import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule

import layout_data.utils.np_transforms as transforms
from layout_data.models.unet import UNet
from layout_data.utils.visualize import visualize_heatmap
from layout_data.data.layout import LayoutDataset
from layout_data.loss.ulloss import Jacobi_layer, Jacobi_layerSoft, OHEMF12d


class UnetUL(LightningModule):
    """
    The implementation of physics-informed CNN for temperature field prediction of heat source layout
    without labeled data
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.model = UNet(in_channels=1, num_classes=1, bn=False)

    def _build_loss(self):
        self.jacobi = Jacobi_layer(nx=self.hparams.nx, length=self.hparams.length, bcs=self.hparams.bcs)

    def forward(self, x):
        y = self.model(x)
        return y

    def __dataloader(self, dataset, batch_size, shuffle=True):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([self.hparams.mean_layout]),
                torch.tensor([self.hparams.std_layout]),
            ),
        ])
        transform_heat = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.train_dir, list_path=self.hparams.train_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        val_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.val_dir, list_path=self.hparams.val_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.test_dir, list_path=self.hparams.test_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )

        print(
            f"Prepared dataset, train:{len(train_dataset)},\
                val:{len(val_dataset)}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)

    def training_step(self, batch, batch_idx):
        layout, _ = batch
        heat_pre = self(layout)

        layout = layout * self.hparams.std_layout + self.hparams.mean_layout
        # The loss of govern equation + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = self.jacobi(layout, heat_pre, 1)

        loss_fun = OHEMF12d(loss_fun=F.l1_loss)
        # loss_fun = torch.nn.MSELoss()
        # loss_fun = torch.nn.L1Loss()
        loss_jacobi = loss_fun(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi))

        loss = loss_jacobi

        self.log('loss_jacobi', loss_jacobi)
        self.log('loss', loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)
        heat_pred_k = heat_pre + 298

        layout = layout * self.hparams.std_layout + self.hparams.mean_layout

        loss_jacobi = F.l1_loss(
            heat_pre, self.jacobi(layout, heat_pre.detach(), 1)
        )
        val_mae = F.l1_loss(heat_pred_k, heat)

        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.1, self.hparams.nx)
            y = np.linspace(0, 0.1, self.hparams.nx)
            visualize_heatmap(x, y, heat_list, heat_pre_list, self.current_epoch)

        return {"val_loss_jacobi": loss_jacobi,
                "val_mae": val_mae}

    def validation_epoch_end(self, outputs):
        val_loss_jacobi_mean = torch.stack([x["val_loss_jacobi"] for x in outputs]).mean()
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()

        self.log('val_loss_jacobi', val_loss_jacobi_mean)
        self.log('val_mae_mean', val_mae_mean)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass


class UnetSL(LightningModule):
    """
    The implementation of supervised vision for comparison.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()

    def _build_model(self):
        self.model = UNet(in_channels=1, num_classes=1, bn=False)

    def forward(self, x):
        y = self.model(x)
        return y

    def __dataloader(self, dataset, batch_size, shuffle=True):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([self.hparams.mean_layout]),
                torch.tensor([self.hparams.std_layout]),
            ),
        ])
        transform_heat = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.train_dir, list_path=self.hparams.train_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        val_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.val_dir, list_path=self.hparams.val_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.test_dir, list_path=self.hparams.test_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )

        print(
            f"Prepared dataset, train:{len(train_dataset)},\
                val:{len(val_dataset)}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)

    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)

        # loss_fun = OHEMF12d(loss_fun=F.l1_loss)
        # loss_fun = torch.nn.MSELoss()
        loss_fun = torch.nn.L1Loss()
        loss = loss_fun(heat_pre, heat - 298.0)

        self.log('loss', loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)
        heat_pred_k = heat_pre + 298

        val_mae = F.l1_loss(heat_pred_k, heat)

        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.1, self.hparams.nx)
            y = np.linspace(0, 0.1, self.hparams.nx)
            visualize_heatmap(x, y, heat_list, heat_pre_list, self.current_epoch)

        return {"val_mae": val_mae}

    def validation_epoch_end(self, outputs):
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()
        self.log('val_mae_mean', val_mae_mean)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass


class UnetULSoft(LightningModule):
    """
    Soft contraints are applied for the boundary conditions.
    """
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.model = UNet(in_channels=1, num_classes=1, bn=False)

    def _build_loss(self):
        self.jacobi = Jacobi_layerSoft(nx=self.hparams.nx, length=self.hparams.length, bcs=self.hparams.bcs)

    def forward(self, x):
        y = self.model(x)
        return y

    def __dataloader(self, dataset, batch_size, shuffle=True):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([self.hparams.mean_layout]),
                torch.tensor([self.hparams.std_layout]),
            ),
        ])
        transform_heat = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.train_dir, list_path=self.hparams.train_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        val_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.val_dir, list_path=self.hparams.val_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.test_dir, list_path=self.hparams.test_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )

        print(
            f"Prepared dataset, train:{len(train_dataset)},\
                val:{len(val_dataset)}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)

    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)

        layout = layout * self.hparams.std_layout + self.hparams.mean_layout
        # The loss of govern equation + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = self.jacobi(layout, heat_pre, 1)

        loss_fun = OHEMF12d(loss_fun=F.l1_loss)
        # loss_fun = torch.nn.MSELoss()
        # loss_fun = torch.nn.L1Loss()
        loss_jacobi = loss_fun(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi))
        loss_D = F.l1_loss(heat_pre[..., 90:110, :1], torch.zeros_like(heat_pre[..., 90:110, :1]))

        loss = loss_jacobi + 0.001 * loss_D

        self.log('loss_jacobi', loss_jacobi)
        self.log('loss_D', loss_D)
        self.log('loss', loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)
        heat_pred_k = heat_pre + 298

        layout = layout * self.hparams.std_layout + self.hparams.mean_layout

        loss_jacobi = F.l1_loss(
            heat_pre, self.jacobi(layout, heat_pre.detach(), 1)
        )
        val_mae = F.l1_loss(heat_pred_k, heat)

        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.1, self.hparams.nx)
            y = np.linspace(0, 0.1, self.hparams.nx)
            visualize_heatmap(x, y, heat_list, heat_pre_list, self.current_epoch)

        return {"val_loss_jacobi": loss_jacobi,
                "val_mae": val_mae}

    def validation_epoch_end(self, outputs):
        val_loss_jacobi_mean = torch.stack([x["val_loss_jacobi"] for x in outputs]).mean()
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()

        self.log('val_loss_jacobi', val_loss_jacobi_mean)
        self.log('val_mae_mean', val_mae_mean)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

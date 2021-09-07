import torch
from torch.utils.data import DataLoader
from layout_data.models.unet import UNet, UNetV2
from layout_data.models.fpn.fpn import fpn
from fourier_neural_operator.fourier_2d import FNO2d
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule
from layout_data.data.layout import LayoutDataset
import layout_data.utils.np_transforms as transforms
from layout_data.loss.ulloss import Jacobi_layer, Jacobi_layerSoft, OHEMF12d, Laplace_ununiform, heat_conservation, MultiScaleLayout
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from layout_data.utils.visualize import visualize_heatmap


class FPNModelUL(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.model = UNet(in_channels=1, num_classes=1, bn=True)

    def _build_loss(self):
        self.jacobi = Jacobi_layer(nx=self.hparams.nx, length=self.hparams.length, bcs=self.hparams.bcs)
        self.heat_conservation = heat_conservation(nx=self.hparams.nx, length=self.hparams.length, bcs=self.hparams.bcs)

    def _build_laplace(self):
        data_path = '/mnt/zhaoxiaoyu/data/layout_data/simple_component/one_point_dataset_200x200/train/Example1.mat'
        data = sio.loadmat(data_path)
        xs, ys = torch.from_numpy(data['xs']).cuda(), torch.from_numpy(data['ys']).cuda()
        self.laplace = Laplace_ununiform(nx=self.hparams.nx, length=self.hparams.length,
                                         bcs=self.hparams.bcs, batch_size=self.hparams.batch_size,
                                         xs=xs, ys=ys)

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
        scheduler = ExponentialLR(optimizer, gamma=0.90)
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
        return self.__dataloader(self.test_dataset, batch_size=16, shuffle=False)

    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)

        # 控制方程损失 + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = self.jacobi(layout * self.hparams.std_layout + self.hparams.mean_layout, heat_pred.detach(), 1)

        ohem_loss1 = OHEMF12d()
        loss_jacobi = ohem_loss1(heat_pred, heat_jacobi)

        # 区域能量守恒损失
        loss_heat_conservation = self.heat_conservation(layout * self.hparams.std_layout + self.hparams.mean_layout, heat_pred)
        loss = loss_jacobi + loss_heat_conservation

        self.log('loss_jacobi', loss_jacobi)
        self.log('loss_heat_conservation', loss_heat_conservation)
        self.log('loss', loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)
        heat_pred_k = heat_pred + 298
        loss_jacobi = F.l1_loss(
            heat_pred, self.jacobi(layout * self.hparams.std_layout + self.hparams.mean_layout, heat_pred.detach(), 1)
        )
        loss_heat_conservation = self.heat_conservation(layout * self.hparams.std_layout + self.hparams.mean_layout, heat_pred)
        val_mae = F.l1_loss(heat_pred_k, heat)

        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.5, self.hparams.nx * 5)
            y = np.linspace(0.1, 0, self.hparams.nx)
            visualize_heatmap(x, y, heat_list, heat_pre_list)

        return {"val_loss_jacobi": loss_jacobi,
                "val_loss_heat_conservation": loss_heat_conservation,
                "val_mae": val_mae}

    def validation_epoch_end(self, outputs):
        val_loss_jacobi_mean = torch.stack([x["val_loss_jacobi"] for x in outputs]).mean()
        val_loss_heat_conservation = torch.stack([x["val_loss_heat_conservation"] for x in outputs]).mean()
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()

        self.log('val_loss_jacobi', val_loss_jacobi_mean)
        self.log('val_loss_heat_conservation', val_loss_heat_conservation)
        self.log('val_mae_mean', val_mae_mean)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass


class UnetMultiScale(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.model = UNetV2(in_channels=1, num_classes=1, bn=False, multi_scale=False)

    def _build_loss(self):
        self.jacobi = Jacobi_layer(nx=self.hparams.nx, length=self.hparams.length, bcs=self.hparams.bcs)

    def _build_laplace(self):
        data_path = '/mnt/zhaoxiaoyu/data/layout_data/simple_component/one_point_dataset_200x200/train/Example1.mat'
        data = sio.loadmat(data_path)
        xs, ys = torch.from_numpy(data['xs']).cuda(), torch.from_numpy(data['ys']).cuda()
        self.laplace = Laplace_ununiform(nx=self.hparams.nx, length=self.hparams.length,
                                         bcs=self.hparams.bcs, batch_size=self.hparams.batch_size,
                                         xs=xs, ys=ys)

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
        # 控制方程损失 + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = self.jacobi(layout, heat_pre, 1)

        ohem_loss1 = OHEMF12d(loss_fun=F.l1_loss)
        # ohem_loss1 = torch.nn.MSELoss()
        # ohem_loss1 = torch.nn.L1Loss()
        loss_jacobi = ohem_loss1(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi))

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


class UnetMultiScaleL(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.model = UNetV2(in_channels=1, num_classes=1, bn=False, multi_scale=False)

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
        layout, heat = batch
        heat_pre = self(layout)

        ohem_loss1 = OHEMF12d(loss_fun=F.l1_loss)
        # ohem_loss1 = torch.nn.MSELoss()
        # ohem_loss1 = torch.nn.L1Loss()
        loss = ohem_loss1(heat_pre, heat - 298.0)

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


class MultiModelUL(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        # self.model = UNetV2(in_channels=1, num_classes=1, bn=False, multi_scale=False)
        self.model = fpn()
        # self.model = FNO2d(50, 50, 64, input_channels=1)

    def _build_loss(self):
        self.jacobi = Jacobi_layer(nx=self.hparams.nx, length=self.hparams.length, bcs=self.hparams.bcs)

    def _build_laplace(self):
        data_path = '/mnt/zhaoxiaoyu/data/layout_data/simple_component/one_point_dataset_200x200/train/Example1.mat'
        data = sio.loadmat(data_path)
        xs, ys = torch.from_numpy(data['xs']).cuda(), torch.from_numpy(data['ys']).cuda()
        self.laplace = Laplace_ununiform(nx=self.hparams.nx, length=self.hparams.length,
                                         bcs=self.hparams.bcs, batch_size=self.hparams.batch_size,
                                         xs=xs, ys=ys)

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
        # 控制方程损失 + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = self.jacobi(layout, heat_pre, 1)

        ohem_loss1 = OHEMF12d(loss_fun=F.l1_loss)
        # ohem_loss1 = torch.nn.MSELoss()
        # ohem_loss1 = torch.nn.L1Loss()
        loss_jacobi = ohem_loss1(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi))

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


class MultiModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        # self.model = UNetV2(in_channels=1, num_classes=1, bn=False, multi_scale=False)
        self.model = fpn()
        # self.model = FNO2d(50, 50, 64, input_channels=1)

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
        layout, heat = batch
        heat_pre = self(layout)

        ohem_loss1 = OHEMF12d(loss_fun=F.l1_loss)
        # ohem_loss1 = torch.nn.MSELoss()
        # ohem_loss1 = torch.nn.L1Loss()
        loss = ohem_loss1(heat_pre, heat - 298.0)

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


class UnetMultiScaleSoft(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        self._build_loss()

    def _build_model(self):
        self.model = UNetV2(in_channels=1, num_classes=1, bn=False, multi_scale=False)

    def _build_loss(self):
        self.jacobi = Jacobi_layerSoft(nx=self.hparams.nx, length=self.hparams.length, bcs=self.hparams.bcs)

    def _build_laplace(self):
        data_path = '/mnt/zhaoxiaoyu/data/layout_data/simple_component/one_point_dataset_200x200/train/Example1.mat'
        data = sio.loadmat(data_path)
        xs, ys = torch.from_numpy(data['xs']).cuda(), torch.from_numpy(data['ys']).cuda()
        self.laplace = Laplace_ununiform(nx=self.hparams.nx, length=self.hparams.length,
                                         bcs=self.hparams.bcs, batch_size=self.hparams.batch_size,
                                         xs=xs, ys=ys)

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
        # 控制方程损失 + Online Hard Sample Mining
        with torch.no_grad():
            heat_jacobi = self.jacobi(layout, heat_pre, 1)

        ohem_loss1 = OHEMF12d(loss_fun=F.l1_loss)
        # ohem_loss1 = torch.nn.MSELoss()
        # ohem_loss1 = torch.nn.L1Loss()
        loss_jacobi = ohem_loss1(heat_pre - heat_jacobi, torch.zeros_like(heat_pre - heat_jacobi))
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
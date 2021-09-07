import torch
from torch.utils.data import DataLoader
from fourier_neural_operator.fourier_2d import FNO2d, FNO2dV2
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule
from layout_data.data.layout import LayoutDataset
import layout_data.utils.np_transforms as transforms
from layout_data.loss.ulloss import Jacobi_layer, OHEMF12d, Laplace_ununiform, heat_conservation, MultiScaleLayout
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from layout_data.utils.visualize import visualize_heatmap


class UnetMultiScale(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()

    def _build_model(self):
        self.modes = 100
        self.width = 32
        self.model = FNO2dV2(self.modes, self.modes, self.width)

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
        scheduler = ExponentialLR(optimizer, gamma=0.80)
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
            transforms.Normalize(
                torch.tensor([298.0]),
                torch.tensor([1.0]),
            ),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.train_dir, list_path=self.hparams.train_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
            max_iters=self.hparams.max_iters,
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
        layout, input, heat = batch

        # loss_fn = OHEMF12d(F.l1_loss)

        output = self(input.permute(0, 2, 3, 1))
        loss = F.l1_loss(output.permute(0, 3, 1, 2), heat)

        self.log('loss', loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, input, heat = batch

        output = self(input.permute(0, 2, 3, 1))
        heat_pred = output.permute(0, 3, 1, 2)
        val_mae = F.l1_loss(heat_pred, heat)

        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred[heat_idx, :, :, :].squeeze().cpu().numpy())
            x = np.linspace(0, 0.5, self.hparams.nx * 5)
            y = np.linspace(0.1, 0, self.hparams.nx)
            visualize_heatmap(x, y, heat_list, heat_pre_list)

        return {"val_mae": val_mae}

    def validation_epoch_end(self, outputs):
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()

        self.log('val_mae_mean', val_mae_mean)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass
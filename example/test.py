"""
Runs a model on a single node across multiple gpus.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import sys
import os
import scipy.io as sio
from layout_data.loss.ulloss import Jacobi_layer

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.models.model import UnetUL
from layout_data.utils.options import parses_ul


def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    model = UnetUL(hparams).cuda()

    model_path = '/mnt/jfs/zhaoxiaoyu/PI-UNet_HSL-TFP/example/lightning_logs/version_0/checkpoints/epoch=29-step=239999.ckpt'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.prepare_data()
    data_loader = model.test_dataloader()
    mae_list, cmae_list, maxae_list, mtae_list, loss_list = [], [], [], [], []
    jacobi = Jacobi_layer(nx=200, length=0.1, bcs=[[[0.045, 0.0], [0.055, 0.0]]])
    for idx, data in enumerate(data_loader):
        model.eval()
        layout, heat = data[0].cuda(), data[1].cuda()
        with torch.no_grad():
            heat_pre = model(layout)

            # Loss
            loss = F.l1_loss(heat_pre, jacobi(layout, heat_pre.detach(), 1))
            loss_list.append(loss.item())

            heat_pre = heat_pre + 298
            # MAE
            mae = F.l1_loss(heat, heat_pre)
            mae_list.append(mae.item())
            # CMAE
            cmae = F.l1_loss(heat, heat_pre, reduction='none').squeeze()
            x_index, y_index = torch.where(layout.squeeze() > 0.0)
            cmae = cmae[x_index, y_index]
            cmae = cmae.mean()
            cmae_list.append(cmae.item())
            # Max AE
            maxae = F.l1_loss(heat, heat_pre, reduction='none').squeeze()
            maxae = torch.max(maxae)
            maxae_list.append(maxae.item())
            # MT AE
            mtae = torch.abs(torch.max(heat_pre) - torch.max(heat))
            mtae_list.append(mtae.item())

        if (idx + 1) % 500 == 0:
            print(idx + 1)

    loss_list = np.array(loss_list)
    mae_list = np.array(mae_list)
    cmae_list = np.array(cmae_list)
    maxae_list = np.array(maxae_list)
    mtae_list = np.array(mtae_list)
    print('-' * 20)
    print('Loss:', loss_list.mean())
    print('MAE:', mae_list.mean())
    print('CMAE:', cmae_list.mean())
    print('Max AE:', maxae_list.mean())
    print('MT AE:', mtae_list.mean())
    sio.savemat('complex_batch_size_1_test.mat',
                {'loss': loss_list, 'mae': mae_list, 'cmae': cmae_list, 'maxae': maxae_list, 'mtae': mtae_list})


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    config_path = Path(__file__).absolute().parent / "config_ul.yml"
    hparams = parses_ul(config_path)
    main(hparams)

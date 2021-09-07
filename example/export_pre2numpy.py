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
from layout_data.models.model import UnetMultiScale
from layout_data.utils.options import parses_ul


def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    model = UnetMultiScale(hparams).cuda()

    model_path = '/mnt/data3/zhaoxiaoyu/layout-data-master/example/lightning_logs/complex_mse/checkpoints/epoch=29-step=239999.ckpt'
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.prepare_data()
    data_loader = model.test_dataloader()
    jacobi = Jacobi_layer(nx=200, length=0.1, bcs=[[[0.045, 0.0], [0.055, 0.0]]])
    for idx, data in enumerate(data_loader):
        model.train()
        layout, heat = data[0].cuda(), data[1].cuda()
        with torch.no_grad():
            heat_pre = model(layout)

            # Loss
            loss = F.l1_loss(heat_pre, jacobi(layout * 10000, heat_pre.detach(), 1), reduction='none')
            loss_heat = F.l1_loss(heat, jacobi(layout * 10000, heat.detach(), 1), reduction='none')

            heat_pre = heat_pre + 298
            # MAE
            mae = F.l1_loss(heat, heat_pre)
            sio.savemat("export_data/complex_" + str(idx) + '_' + str(mae.item()) + '.mat', {
                'layout': (layout * 10000).cpu().squeeze().numpy(),
                'u': heat.cpu().squeeze().numpy(),
                'u_pre': heat_pre.detach().cpu().squeeze().numpy(),
                'loss': loss.cpu().squeeze().numpy(),
                'loss_heat': loss_heat.cpu().squeeze().numpy()
            })
        if idx > 4:
            break


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    config_path = Path(__file__).absolute().parent / "config_ul.yml"
    hparams = parses_ul(config_path)
    main(hparams)

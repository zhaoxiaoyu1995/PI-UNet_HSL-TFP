from pathlib import Path
import numpy as np
import torch
from torch.backends import cudnn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.utils.options import parses_ul
from layout_data.models.model import UnetSL


def main(hparams):
    """
    Main training routine specific for this project
    """
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = UnetSL(hparams)
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_mae_mean', mode='min')
    trainer = pl.Trainer(
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gpus=[hparams.gpus],
        precision=16 if hparams.use_16bit else 32,
        val_check_interval=hparams.val_check_interval,
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        profiler=hparams.profiler,
        weights_summary=None,
        benchmark=True,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    print(hparams)
    print()
    trainer.fit(model)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    config_path = Path(__file__).absolute().parent / "config_ul.yml"
    hparams = parses_ul(config_path)
    main(hparams)

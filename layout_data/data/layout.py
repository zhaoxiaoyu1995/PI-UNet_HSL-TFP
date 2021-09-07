# -*- encoding: utf-8 -*-
"""
Desc      :   Layout dataset
"""
# File    :   layout.py
# Time    :   2020/05/26 10:28:04
# Author  :   Zweien
# Contact :   278954153@qq.com

import os
from .loadresponse import LoadResponse, mat_loader, LoadResponseH5


class LayoutDataset(LoadResponse):
    """Layout dataset (mutiple files) generated by 'layout-generator'.
    """

    def __init__(
        self,
        root,
        list_path,
        subdir,
        transform=None,
        target_transform=None,
        load_name="list",
        resp_name="u",
        max_iters=None,
        nx=200,
    ):
        root = os.path.join(root, subdir)
        super().__init__(
            root,
            mat_loader,
            list_path,
            load_name=load_name,
            resp_name=resp_name,
            extensions="mat",
            transform=transform,
            target_transform=target_transform,
            max_iters=max_iters,
            nx=nx,
        )


class LayoutDatasetH5(LoadResponseH5):
    """Layout dataset (hdf5 format, single file) generated by 'layout-generator'.
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        train_fn="train.h5",
        test_fn="test.h5",
        train_dir="./",
        test_dir="./",
        load_name="F",
        resp_name="u",
    ):

        fn = train_fn if train else test_fn
        if train:
            fn = os.path.join(train_dir, train_fn)
        else:
            fn = os.path.join(test_dir, test_fn)
        root = os.path.join(root, fn)
        super().__init__(
            root,
            load_name=load_name,
            resp_name=resp_name,
            transform=transform,
            target_transform=target_transform,
        )
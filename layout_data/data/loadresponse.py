# -*- encoding: utf-8 -*-
"""
Desc      :   Load Response Dataset.
"""
# File    :   loadresponse.py
# Time    :   2020/04/06 17:24:13
# Author  :   Zweien
# Contact :   278954153@qq.com

import os
import scipy.io as sio
import h5py
import numpy as np
from torchvision.datasets import VisionDataset


class LoadResponse(VisionDataset):
    """Some Information about LoadResponse dataset"""

    def __init__(
        self,
        root,
        loader,
        list_path,
        load_name="F",
        resp_name="u",
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        max_iters=None,
        nx=200,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.list_path = list_path
        self.loader = loader
        self.load_name = load_name
        self.resp_name = resp_name
        self.nx = nx
        self.extensions = extensions
        self.sample_files = make_dataset_list(root, list_path, extensions, is_valid_file, max_iters=max_iters)

    def __getitem__(self, index):
        path = self.sample_files[index]
        load, resp = self.loader(path, self.load_name, self.resp_name, self.nx)
        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        return load, resp

    def __len__(self):
        return len(self.sample_files)


class LoadResponseH5(VisionDataset):
    def __init__(
        self,
        root,
        load_name="F",
        resp_name="u",
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root, transform=transform, target_transform=target_transform,
        )
        self.load_name = load_name
        self.resp_name = resp_name
        self.data_info = self._get_info(root)

    def _get_info(self, path):
        """get h5 info
        """
        data_info = {}
        with h5py.File(path, "r") as file:
            for key, value in file.items():
                _len, *shape = value.shape
                data_info[key] = {"len": _len, "shape": shape}
        return data_info

    def __getitem__(self, index):
        with h5py.File(self.root, "r") as file:
            load = file[self.load_name][index]
            resp = file[self.resp_name][index]
        if self.transform is not None:
            load = self.transform(load)
        if self.target_transform is not None:
            resp = self.target_transform(resp)
        return load, resp

    def __len__(self):
        return self.data_info[self.load_name]['len']


def make_dataset(root_dir, extensions=None, is_valid_file=None):
    """make_dataset() from torchvision.
    """
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    for root, _, fns in sorted(os.walk(root_dir, followlinks=True)):
        for fn in sorted(fns):
            path = os.path.join(root, fn)
            if is_valid_file(path):
                files.append(path)
    return files


def make_dataset_list(root_dir, list_path, extensions=None, is_valid_file=None, max_iters=None):
    """make_dataset() from torchvision.
        """
    files = []
    root_dir = os.path.expanduser(root_dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file \
                cannot be None or not None at the same time"
        )
    if extensions is not None:
        is_valid_file = lambda x: has_allowed_extension(x, extensions)

    assert os.path.isdir(root_dir), root_dir
    with open(list_path, 'r') as rf:
        for line in rf.readlines():
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            if is_valid_file(path):
                files.append(path)
    if max_iters is not None:
        files = files * int(np.ceil(float(max_iters) / len(files)))
        files = files[:max_iters]
    return files


def has_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def mat_loader(path, load_name, resp_name=None, nx=200):
    mats = sio.loadmat(path)
    if load_name == 'F':
        load = mats.get(load_name).astype(np.float32)
    elif load_name == 'list':
        load = mats.get(load_name)[0]
        layout_map = np.zeros((nx, nx))
        mul = int(nx / 10)
        for i in load:
            i = i - 1
            layout_map[(i % 10 * mul):((i % 10) * mul + mul), (i // 10 * mul):((i // 10 * mul) + mul)] = 10000 * np.ones((mul, mul))
        load = layout_map
    resp = mats.get(resp_name) if resp_name is not None else None
    return load, resp


if __name__ == "__main__":
    total_num = 50000
    with open('train'+str(total_num)+'.txt', 'w') as wf:
        for idx in range(int(total_num*0.8)):
            wf.write('Example'+str(idx)+'.mat'+'\n')
    with open('val'+str(total_num)+'.txt', 'w') as wf:
        for idx in range(int(total_num*0.8), total_num):
            wf.write('Example'+str(idx)+'.mat'+'\n')
from pathlib import Path
import scipy.io as sio
import h5py


def load_mat(path):
    path = Path(path)
    assert path.suffix == ".mat"
    return sio.loadmat(path)


def load_h5(path):
    h5file = h5py.File(path, "r")
    return h5file

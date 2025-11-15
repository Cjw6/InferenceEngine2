import numpy as np


def check_npy_data(path1: str, path2: str):
    data1 = np.load(path1)
    data2 = np.load(path2)
    print(np.allclose(data1, data2))


if __name__ == "__main__":
    check_npy_data("random_data.npy", "random_data_loaded.npy")

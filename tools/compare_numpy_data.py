import numpy as np
import sys


def safe_load_data(file_path: str):
    data = np.load(file_path)
    data = data.astype(np.float32)
    return data


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_numpy_data.py <file_path1> <file_path2>")

    data1 = safe_load_data(sys.argv[1])
    data2 = safe_load_data(sys.argv[2])

    if np.allclose(data1, data2, rtol=1e-05, atol=1e-08, equal_nan=False):
        print("The data is the same.")
    else:
        print("The data is different.")

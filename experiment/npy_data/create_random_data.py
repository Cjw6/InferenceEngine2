import numpy as np


def create_random_data():
    print("create_random_data")
    data = np.random.rand(20, 30, 40, 50)
    data = data.astype(np.float32)
    np.save("random_data.npy", data)


if __name__ == "__main__":
    create_random_data()

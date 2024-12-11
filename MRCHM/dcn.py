import numpy as np
from scipy.fft import dct


def dcn(raw_data, target_rate=24000):
    data = dct(raw_data)
    if len(data) < target_rate:
        data = np.pad(data, (0, target_rate - len(data)))
    else:
        data = data[:target_rate]
    data_energy = np.sum(data ** 2)
    data = data * np.sqrt(target_rate / data_energy) / 100
    data = data.astype(np.float32)
    return data


if __name__ == '__main__':
    rand_data = np.random.rand(123456)
    print(dcn(rand_data).shape)
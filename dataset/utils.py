import numpy as np


def array_yxc2cyx(arr):
    arr = np.swapaxes(arr, 1, 2)
    arr = np.swapaxes(arr, 0, 1)
    return arr


def array_cyx2yxc(arr):
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr


def normalize_cv_img(cv_img, mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)):
    r_mean, g_mean, b_mean = mean
    r_std, g_std, b_std = std
    img = cv_img.astype(np.float32)
    img = cv_img / 255.
    out = np.zeros(shape=img.shape, dtype=np.float32)
    out[:, :, 0] = (img[:, :, 0] - b_mean) / b_std
    out[:, :, 1] = (img[:, :, 1] - g_mean) / g_std
    out[:, :, 2] = (img[:, :, 2] - r_mean) / r_std
    return out


def denormalize_cv_array(cv_arr, mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)):
    r_mean, g_mean, b_mean = mean
    r_std, g_std, b_std = std
    out = np.zeros(shape=cv_arr.shape, dtype=np.float32)
    out[:, :, 0] = cv_arr[:, :, 0] * b_std + b_mean
    out[:, :, 1] = cv_arr[:, :, 1] * g_std + g_mean
    out[:, :, 2] = cv_arr[:, :, 2] * r_std + r_mean
    out = out * 255.
    return out


def encode_img(image, mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)):
    norm = normalize_cv_img(image, mean=mean, std=std)
    return array_yxc2cyx(norm)


def decode_img(tensor, mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)):
    arr = array_cyx2yxc(tensor)
    arr = denormalize_cv_array(arr, mean=mean, std=std)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


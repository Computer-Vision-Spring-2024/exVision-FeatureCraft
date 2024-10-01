import numpy as np


def convert_to_gray(img_RGB: np.ndarray) -> np.ndarray:
    if len(img_RGB.shape) == 3:
        grey = np.dot(img_RGB[..., :3], [0.2989, 0.5870, 0.1140])
        return grey.astype(np.uint8)
    else:
        return img_RGB.astype(np.uint8)


def convert_BGR_to_RGB(img_BGR_nd_arr: np.ndarray) -> np.ndarray:
    img_RGB_nd_arr = img_BGR_nd_arr[..., ::-1]
    return img_RGB_nd_arr

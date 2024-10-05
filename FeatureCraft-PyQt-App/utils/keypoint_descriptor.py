from math import cos, sin

import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.transform import rescale, resize

from utils.SIFT_scale_space import *


def sift_resize(img, ratio=None):
    """
    Resize an image while maintaining its aspect ratio.

    Parameters:
    - img (numpy.ndarray): The input image to be resized.
    - ratio (float, optional): The ratio by which the image should be resized. If None, it is calculated
    based on the square root of (1024*1024) divided by the product of the input image's width and height.

    Returns:
    - resized_img (numpy.ndarray): The resized image.
    - ratio (float): The ratio used for resizing the image.

    Notes:
    - The `resize` function used here resizes the image to the new shape calculated based on the ratio.
    - `anti_aliasing=True` is used to smooth the edges of the resized image.
    """
    ratio = (
        ratio if ratio is not None else np.sqrt((1024 * 1024) / np.prod(img.shape[:2]))
    )
    newshape = list(map(lambda d: int(round(d * ratio)), img.shape[:2]))
    img = resize(img, newshape, anti_aliasing=True)
    return img, ratio


def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image


def represent_keypoints(keypoints, DoG):
    """
    Represent keypoints as boolean images indicating their presence in different levels of the Difference of Gaussian (DoG) pyramid.

    Parameters:
    - keypoints (list): A list of lists containing keypoints for each octave. Each keypoint is represented as a tuple (x, y, sigma),
                        where x and y are the coordinates of the keypoint and sigma is the scale at which it was detected.
    - DoG (list): A list of Difference of Gaussian (DoG) images for each octave. Each octave contains a series of images
                representing the difference between blurred images at different scales.

    Returns:
    - keypoints_as_images (list): A list of boolean images representing the presence of keypoints at different scales
                                within each octave of the DoG pyramid. Each element in the list corresponds to an octave,
                                and contains boolean images indicating keypoints detected at different levels of the DoG pyramid.
    """
    keypoints_as_images = list()
    for octave_ind, kp_per_octave in enumerate(keypoints):
        keypoints_per_octave = list()
        for dog_idx in range(1, len(DoG) - 1):  # the boundaries are not included
            keypoints_per_sigma = np.full(
                DoG[octave_ind][0].shape, False, dtype=bool
            )  # create bool 2d array of same size as
            for kp in kp_per_octave:
                if kp[2] == dog_idx:
                    keypoints_per_sigma[kp[0], kp[1]] = True
            keypoints_per_octave.append(keypoints_per_sigma)
        keypoints_as_images.append(keypoints_per_octave)
    return keypoints_as_images


def sift_gradient(img):
    dx = np.array([-1, 0, 1]).reshape((1, 3))
    dy = dx.T
    gx = convolve2d(img, dx, boundary="symm", mode="same")
    gy = convolve2d(img, dy, boundary="symm", mode="same")
    magnitude = np.sqrt(gx * gx + gy * gy)
    direction = np.rad2deg(np.arctan2(gy, gx)) % 360  # to wrap the direction
    return gx, gy, magnitude, direction


def padded_slice(img, sl):
    """
    Extract a slice from the input image with padding to match the specified output shape.

    Parameters:
    - img (numpy.ndarray): Input image.
    - sl (list): List containing slice indices [start_row, end_row, start_column, end_column].

    Returns:
    - output (numpy.ndarray): Padded slice of the input image based on the specified slice indices.

    Notes:
    - The function extracts a slice from the input image based on the specified slice indices.
    - If the slice extends beyond the boundaries of the image, padding is applied to match the specified output shape.
    - The output shape is determined by the difference between the end and start indices of the slice.
    - Padding is applied using zero values.
    """
    output_shape = np.asarray(np.shape(img))
    output_shape[0] = sl[1] - sl[0]
    output_shape[1] = sl[3] - sl[2]
    src = [
        max(sl[0], 0),
        min(sl[1], img.shape[0]),
        max(sl[2], 0),
        min(sl[3], img.shape[1]),
    ]
    dst = [src[0] - sl[0], src[1] - sl[0], src[2] - sl[2], src[3] - sl[2]]
    output = np.zeros(
        output_shape, dtype=img.dtype
    )  # padding of zeros if the indices of sl is out of the image boundaries
    output[dst[0] : dst[1], dst[2] : dst[3]] = img[src[0] : src[1], src[2] : src[3]]
    return output


def dog_keypoints_orientations(img_gaussians, keypoints, sigma_base, num_bins=36, s=2):
    """Assigns the dominant orientation of the keypoint"""

    kps = []
    for octave_idx in range(len(img_gaussians)):  # iterate over the ocataves
        img_octave_gaussians = img_gaussians[octave_idx]
        octave_keypoints = keypoints[octave_idx]
        for idx, scale_keypoints in enumerate(octave_keypoints):
            scale_idx = (
                idx + 1
            )  ## This will be adjusted according to the sigma surface resulting from interpolation. (skip for now)
            gaussian_img = img_octave_gaussians[scale_idx]
            sigma = (
                1.5
                * sigma_base
                * (2**octave_idx)
                * (
                    (2 ** (1 / s)) ** (scale_idx)
                )  # sigma for smoothing the magnitude accordingly (1.5 recommmended)
            )

            kernel = gaussian_filter_kernel(sigma)
            radius = int(round(sigma * 2))  # 2 x std == 95 %
            gx, gy, magnitude, direction = sift_gradient(gaussian_img)
            direction_idx = np.round(direction * num_bins / 360).astype(
                int
            )  # dirction in terms of bins

            for i, j in map(
                tuple, np.argwhere(scale_keypoints).tolist()
            ):  # get the coordinates of the point
                window = [
                    i - radius,
                    i + radius + 1,
                    j - radius,
                    j + radius + 1,
                ]  # the indices of the window to be extracted
                mag_win = padded_slice(magnitude, window)
                dir_idx = padded_slice(direction_idx, window)
                weight = (
                    mag_win * kernel
                )  # modulate the weights according to the sigma * 1.5 (sigma at which the keypoint is detected)
                hist = np.zeros(num_bins, dtype=np.float32)

                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum(
                        weight[dir_idx == bin_idx]
                    )  # histogram is mag weighted

                for bin_idx in np.argwhere(
                    hist >= 0.8 * hist.max()
                ).tolist():  #  returns list of lists
                    angle = (bin_idx[0] + 0.5) * (360.0 / num_bins) % 360
                    kps.append(
                        (i, j, octave_idx, scale_idx, angle)
                    )  # there can be more than one descriptor to the same keypoint (another dominant angle)
    return kps


def rotated_subimage(image, center, theta, width, height):
    """
    Rotate a subimage around a specified center point by a given angle.

    Parameters:
    - image (numpy.ndarray): Input image.
    - center (tuple): Coordinates (x, y) of the center point around which to rotate the subimage.
    - theta (float): Angle of rotation in degrees.
    - width (int): Width of the subimage.
    - height (int): Height of the subimage.

    Returns:
    - rotated_image (numpy.ndarray): Rotated subimage.

    Notes:
    - The function rotates the subimage around the specified center point by the given angle.
    - Rotation angle `theta` is provided in degrees and converted to radians internally for computation.
    - The function uses an affine transformation to perform the rotation.
    - Nearest-neighbor interpolation is used (`cv2.INTER_NEAREST`) to avoid interpolation artifacts.
    - The `cv2.WARP_INVERSE_MAP` flag indicates that the provided transformation matrix is the inverse transformation matrix.
    - Pixels outside the image boundaries are filled with a constant value (0) using `cv2.BORDER_CONSTANT` border mode.
    """
    theta *= 3.14159 / 180  # convert to rad

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width - 1) / 2) - v_y[0] * ((height - 1) / 2)
    s_y = center[1] - v_x[1] * ((width - 1) / 2) - v_y[1] * ((height - 1) / 2)

    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(
        image,
        mapping,
        (width, height),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
    )


def get_gaussian_mask(sigma, filter_size):
    if sigma > 0:
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - (filter_size - 1) / 2) ** 2 + (y - (filter_size - 1) / 2) ** 2)
                / (2 * sigma**2)
            ),
            (filter_size, filter_size),
        )
        return kernel / np.sum(kernel)
    else:
        raise ValueError("Invalid value of Sigma")


def extract_sift_descriptors(img_gaussians, keypoints, base_sigma, num_bins=8, s=2):
    """Extract the 128 length descriptors of each keypoint besides their keypoint info (i ,j , oct_idx, scale_idx, orientation)"""

    descriptors = []
    points = []
    data = {}  #
    for i, j, oct_idx, scale_idx, orientation in keypoints:

        if "index" not in data or data["index"] != (oct_idx, scale_idx):
            data["index"] = (oct_idx, scale_idx)
            gaussian_img = img_gaussians[oct_idx][
                scale_idx
            ]  # must be editted in case of taylor approximation
            sigma = (
                1.5
                * base_sigma
                * (2**oct_idx)
                * (
                    (2 ** (1 / s)) ** (scale_idx)
                )  # scale invarance introduced to the keypoint (kernel std proportional to sigma of keypoint)
            )
            data["kernel"] = get_gaussian_mask(
                sigma=sigma, filter_size=16
            )  # the window size is constant

            gx, gy, magnitude, direction = sift_gradient(gaussian_img)
            data["magnitude"] = magnitude
            data["direction"] = direction

        window_mag = rotated_subimage(
            data["magnitude"], (j, i), orientation, 16, 16
        )  # rotation to align with the domianant orientation
        window_mag = window_mag * data["kernel"]
        window_dir = rotated_subimage(data["direction"], (j, i), orientation, 16, 16)
        window_dir = (((window_dir - orientation) % 360) * num_bins / 360.0).astype(
            int
        )  # subtract the dominant orientation to make it direction invariance

        features = []  # store the hist of 16 regions concatenated (128)
        for sub_i in range(4):
            for sub_j in range(4):
                sub_weights = window_mag[
                    sub_i * 4 : (sub_i + 1) * 4, sub_j * 4 : (sub_j + 1) * 4
                ]
                sub_dir_idx = window_dir[
                    sub_i * 4 : (sub_i + 1) * 4, sub_j * 4 : (sub_j + 1) * 4
                ]
                hist = np.zeros(num_bins, dtype=np.float32)
                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum(sub_weights[sub_dir_idx == bin_idx])
                features.extend(hist.tolist())
        features = np.array(features)
        features /= np.linalg.norm(features)  # normalize
        np.clip(
            features, np.finfo(np.float16).eps, 0.2, out=features
        )  # clip to remove non-linear illumnation effect (0.2) as descripted by autho
        features /= np.linalg.norm(features)  # renormalize
        descriptors.append(features)
        points.append((i, j, oct_idx, scale_idx, orientation))
    return points, descriptors


def computeKeypointsAndDescriptors(
    image, n_octaves, s_value, sigma_base, constract_th, r_ratio
):
    grayscaled_image = convert_to_grayscale(image)  # convert to grayscale
    base_image = rescale(
        grayscaled_image, 2, anti_aliasing=False
    )  # upsampling to increase the number of features extracted
    pyramid, DoG, keypoints = generate_octaves_pyramid(
        base_image, n_octaves, s_value, sigma_base, constract_th, r_ratio
    )
    keypoints = represent_keypoints(
        keypoints, DoG
    )  # represent the keypoints in each (octave, scale) as bool images
    keypoints_ijso = dog_keypoints_orientations(
        pyramid, keypoints, sigma_base, 36, s_value
    )  # ( i ,j , oct_idx, scale_idx, orientation)
    points, descriptors = extract_sift_descriptors(
        pyramid, keypoints_ijso, sigma_base, 8, s_value
    )
    return points, descriptors


def kp_list_2_opencv_kp_list(kp_list):
    """represnet the keypoints as keyPoint objects"""

    opencv_kp_list = []
    for kp in kp_list:
        opencv_kp = cv2.KeyPoint(
            x=kp[1] * (2 ** (kp[2] - 1)),
            y=kp[0] * (2 ** (kp[2] - 1)),
            size=kp[3],
            angle=kp[4],
        )
        opencv_kp_list += [opencv_kp]

    return opencv_kp_list


def match(img_a, pts_a, desc_a, img_b, pts_b, desc_b, tuning_distance=0.3):
    img_a, img_b = tuple(map(lambda i: np.uint8(i * 255), [img_a, img_b]))

    desc_a = np.array(desc_a, dtype=np.float32)
    desc_b = np.array(desc_b, dtype=np.float32)

    pts_a = kp_list_2_opencv_kp_list(pts_a)
    pts_b = kp_list_2_opencv_kp_list(pts_b)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(
        desc_a, desc_b, k=2
    )  # apply nearest neighbour to get the nearest 2 for each descriptor.
    # Apply ratio test
    good = []
    for m, n in matches:
        if (
            m.distance < tuning_distance * n.distance
        ):  # (if evaluate to "false", then there is confusion around this descriptor, so neglect)
            good.append(m)

    img_match = np.empty(
        (max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3),
        dtype=np.uint8,
    )

    cv2.drawMatches(
        img_a,
        pts_a,
        img_b,
        pts_b,
        good,
        outImg=img_match,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return img_match

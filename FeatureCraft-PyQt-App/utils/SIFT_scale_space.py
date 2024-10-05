import numpy as np
from scipy.signal import convolve2d


def gaussian_filter_kernel(sigma, kernel_size=None):
    """
    Description:
        - Generates a Gaussian filter kernel.

    Args:
        - kernel_size: Size of the square kernel (e.g., 3x3).
        - sigma: Standard deviation of the Gaussian distribution.

    Returns:
        - A numpy array representing the Gaussian filter kernel.
    """

    kernel_size = (4 * sigma) + 1

    offset = kernel_size // 2

    x = np.arange(-offset, offset + 1)[:, np.newaxis]
    y = np.arange(-offset, offset + 1)
    x_squared = x**2
    y_squared = y**2

    kernel = np.exp(-(x_squared + y_squared) / (2 * sigma**2))
    kernel /= 2 * np.pi * (sigma**2)  # for normalization

    return kernel


def generateGaussianKernels(sigma, s):
    """
    Description:
        - Generates the required Gaussian Kernels for generating different scales for each octave.

    Args:
        - sigma: Standard deviation of the Gaussian distribution.
        - num_intervals: the order of the image that is blurred with 2 sigma

    Returns:
        - gaussian_kernels: A numpy array of arrays in which the generated Gaussian kernels are stored.
    """
    gaussian_kernels = []
    scale_level = sigma
    # To cover a complete octave, we need 's + 3' blurred images. This ensures that we have enough information for accurate feature detection.
    images_per_octave = s + 3
    # constant multiplicative factor k which separates two nearby scales in the octave
    k = 2 ** (1 / s)
    gaussian_kernels.append(gaussian_filter_kernel(sigma))
    # generate kernel for each image in the octave
    for iterator in range(1, images_per_octave):
        # multiply the current scale level with the multiplicative factor
        scale_level *= k
        gaussian_kernels.append(gaussian_filter_kernel(scale_level))
    return gaussian_kernels


def get_keypoints(DOG_octave, k, contrast_th, ratio_th, DoG_full_array):
    """
    Description:
        - from each three difference of gaussians images, detect possible keypoints through extrema detection which is done by comparing the middle pixel with
            its eight neighbors in the middle image and nine neighbors in the scale above and below it, then applying keypoints localization
            and filtering to discarde unstable keypoints
    Args:
        - DOG_octave: the last three difference of gaussians calculated.
        - k: the depth of the center pixel in the difference of gaussians array.
        - contrust_th: the minimum contrust threshold for a stable keypoint
        - ratio_th: the maximum threshold for the ratio of principal curvatures, if the ratio exceeds this threshold that indicates that the
            keypoint is an edge, and since that edges can't be keypoints the keypoint is discarded.

    Returns:
        - keypoints: A numpy array of arrays in which the keypoints of the current octave are stored.
    """
    keypoints = []
    # stack the last three difference of gaussians along the depth dimention
    DoG = np.concatenate(
        [o[:, :, np.newaxis] for o in DOG_octave], axis=2
    )  # 2d -- > 3d
    # loop over the middle image and form a 3*3*3 patch
    for i in range(1, DoG.shape[0] - 2):
        for j in range(1, DoG.shape[1] - 2):
            # form a (3*3*3)patch: 3 rows from i-1 to i+1, three columns from j-1 to j+1 and the depth of DoG stack is already three
            patch = DoG[i - 1 : i + 2, j - 1 : j + 2, :]
            # flatten the 27 values of the patch, get the index of the maximum and minimum values of the flattened array, since the total length is 27
            # then the middle pixel index is 13 so if the returned index is 13 then the center pixel is an extrema
            if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                # # localize the detected keypoint
                # # offset, J, H, x, y, s = localize_keypoint(DoG_full_array, j, i, k )
                # if np.max(offset) > 0.5: continue
                # # calculate its contrast
                # contrast = DoG[y,x,s] + 0.5*J.dot(offset)
                # # if the contrast is below the threshold move to the next patch
                # if abs(contrast) < contrast_th: continue
                # tr = H[0][0] + H[1][1]
                # det = H[0][0] * H[1][1] - H[0][1] ** 2
                # r = ( tr ** 2 ) / det
                # # If this ratio is above a certain threshold then the keypoint is an edge therefore skip it and move to the next patch
                # if r > ratio_th: continue
                # # add the final offset to the location of the keypoint to get the interpolated estimate for the location of the keypoint.
                # kp = np.array([x, y, s]) + offset
                # append the keypoint location to the keypoints of the octave
                kp = np.array([i, j, k])
                keypoints.append(kp)
    return np.array(keypoints)


def generate_gaussian_images_in_octave(
    image, gaussian_kernels, contrast_th, ratio_th, octave_index
):
    """
    Description:
        - Generates the octave's increasingly blurred images.

    Args:
        - image: the base image for the octave.
        - gaussian_kernels: A numpy array of arrays in which the generated Gaussian kernels are stored.
        - contrust_th: the minimum contrust threshold for a stable keypoint
        - ratio_th: the maximum threshold for the ratio of principal curvatures, if the ratio exceeds this threshold that indicates that the
            keypoint is an edge, and since that edges can't be keypoints the keypoint is discarded.

    Returns:
        - gaussian_images_in_octave: A numpy array of arrays in which the generated blurred octave images are stored.
        - np.concatenate([o[:,:,np.newaxis] for o in DOG_octave], axis=2): 3D array representing the Difference of gaussians stacked together along the depth dimension.
        - keypoints: A numpy array of arrays in which the keypoints of the current octave are stored.
    """
    # pad the image to perserve its size in the octave
    # array of all the increasingly blurred images per octave
    gaussian_images_in_octave = []
    if octave_index == 0:
        base_image = convolve2d(image, gaussian_kernels[0], "same", "symm")
        # append the first gaussian filtered image to the octave images
        gaussian_images_in_octave.append(base_image)
        # array to store the difference of each two adjacent gaussian filtered image in the octave
    else:
        gaussian_images_in_octave.append(image)

    DOG_octave = []
    # octave keypoints
    keypoints = []
    for gaussian_kernel in gaussian_kernels[1:]:
        # convolve the gaussian kernels with the octave base padded image
        blurred_image = convolve2d(image, gaussian_kernel, "same", "symm")
        gaussian_images_in_octave.append(blurred_image)
        # subtract each two adjacent images and add the result to the difference of gaussians of the octave
        DOG_octave.append(gaussian_images_in_octave[-1] - gaussian_images_in_octave[-2])
        if len(DOG_octave) > 2:
            # from each three difference of gaussians images, detect possible keypoints through extrema detection then applying keypoints localization
            # and filtering to discarde unstable keypoints
            keypoints.extend(
                get_keypoints(
                    DOG_octave[-3:],
                    len(DOG_octave) - 2,
                    contrast_th,
                    ratio_th,
                    np.concatenate([o[:, :, np.newaxis] for o in DOG_octave], axis=2),
                )
            )
    return gaussian_images_in_octave, DOG_octave, keypoints


def generate_octaves_pyramid(
    img, num_octaves=4, s_value=2, sigma=1.6, contrast_th=0.03, ratio_th=10
):
    """
    Description:
        - Generates the gaussian pyramid which consists of several octaves with increasingly blurred images.

    Args:
        - img: the image whose features should be extracted.
        - contrust_th: the minimum contrust threshold for a stable keypoint
        - ratio_th: the maximum threshold for the ratio of principal curvatures, if the ratio exceeds this threshold that indicates that the
            keypoint is an edge, and since that edges can't be keypoints the keypoint is discarded.
        - num_octaves: the number of octaves in the pyramid
        - intervals: the order of the image that is blurred with 2 sigma in the octave

    Returns:
        - gaussian_images_pyramid: A numpy array of arrays in which the generated octaves are stored.
        - DOG_pyramid: A numpy array of arrays in which the difference of gaussians of all octaves are stored.
        - keypoints: A numpy array of arrays in which the keypoints of all octaves are stored.
    """

    # generate the kernels required for generating images per octave
    gaussian_kernels = generateGaussianKernels(
        sigma, s_value
    )  # intervals == s  ( s + 3 )
    # The pyramid of different octaves
    gaussian_images_pyramid = []
    # the pyramid of the difference of gaussians of different octaves
    DOG_pyramid = []
    # the keypoints for all octaves
    keypoints = []
    for octave_index in range(num_octaves):
        # calculate the blurred images of the current octave and the keypoints in the octave and the difference of gaussians which is the subtraction
        # of each two adjacent gaussian filtered images in the octave
        gaussian_images_in_octave, DOG_octave, keypoints_per_octave = (
            generate_gaussian_images_in_octave(
                img, gaussian_kernels, contrast_th, ratio_th, octave_index
            )
        )
        # append the current octave to the pyramid of octaves
        gaussian_images_pyramid.append(gaussian_images_in_octave)
        # append the difference of gaussians images of the current octave to the different of gaussians pyramid
        DOG_pyramid.append(DOG_octave)
        # append the keypoints of the current octave to the keypoints array
        keypoints.append(keypoints_per_octave)
        # Downsample the image that is blurred by 2 sigma to be the base image for the next octave
        img = gaussian_images_in_octave[-3][::2, ::2]
    return gaussian_images_pyramid, DOG_pyramid, keypoints


def localize_keypoint(D, x, y, s):
    """
    Description:
        - refining the detected keypoints to sub-pixel accuracy. This is done by fitting a 3D quadratic function
            to the nearby data to determine the interpolated location of the maximum, In SIFT the second-order Taylor expansion of the DoG octave is used

    Args:
        - D: difference of gaussians stacked along the depth dimention
        - x: the x coordinate of the keypoint.
        - y: the y coordinate of the keypoint
        - s: the depth of the keypoint.

    Returns:
        - offset: the final offset that should be added to the location of the keypoint to get the interpolated estimate for the location of the keypoint
        - J: the first derivatives of D, These derivatives represent the rate of change of difference of gaussians intensity in each direction.
        - H[:2,:2]: the second derivatives (Hessian matrix) of the image intensity at the specified point.
            The Hessian matrix represents the local curvature or second-order rate of change of image intensity.
        - x: the x coordinate of the keypoint after further localization.
        - y: the y coordinate of the keypoint after further localization.
        - s: the depth of the keypoint after further localization..
    """
    # convert D to larger data type (float) to avoid overflow
    D = D.astype(np.float64)
    # computes the first derivatives (gradient) of the image intensity along the x, y, and scale dimensions at the specified point (x, y, s).
    dx = (D[y, x + 1, s] - D[y, x - 1, s]) / 2.0
    dy = (D[y + 1, x, s] - D[y - 1, x, s]) / 2.0
    ds = (D[y, x, s + 1] - D[y, x, s - 1]) / 2.0
    # computes the second derivatives (Hessian matrix) of the image intensity at the keypoint.
    dxx = D[y, x + 1, s] - 2 * D[y, x, s] + D[y, x - 1, s]
    dxy = (
        (D[y + 1, x + 1, s] - D[y + 1, x - 1, s])
        - (D[y - 1, x + 1, s] - D[y - 1, x - 1, s])
    ) / 4
    dxs = (
        (D[y, x + 1, s + 1] - D[y, x - 1, s + 1])
        - (D[y, x + 1, s - 1] - D[y, x - 1, s - 1])
    ) / 4
    dyy = D[y + 1, x, s] - 2 * D[y, x, s] + D[y - 1, x, s]
    dys = (
        (D[y + 1, x, s + 1] - D[y - 1, x, s + 1])
        - (D[y + 1, x, s - 1] - D[y - 1, x, s - 1])
    ) / 4
    dss = D[y, x, s + 1] - 2 * D[y, x, s] + D[y, x, s - 1]
    J = np.array([dx, dy, ds])
    # the second derivatives (Hessian matrix) of the image intensity at the specified point.
    H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    # the final offset that should be added to the location of the keypoint to get the interpolated estimate for the location of the keypoint
    offset = -np.linalg.inv(H).dot(J)  # ((3 x 3) . 3 x 1)
    return offset, J, H[:2, :2], x, y, s

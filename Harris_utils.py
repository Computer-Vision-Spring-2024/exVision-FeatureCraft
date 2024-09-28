import numpy as np 


def convert_to_gray(img_RGB: np.ndarray) -> np.ndarray:
    if len(img_RGB.shape) == 3:
        grey = np.dot(img_RGB[..., :3], [0.2989, 0.5870, 0.1140])
        return grey.astype(np.uint8)
    else:
        return img_RGB.astype(np.uint8)



def padding_matrix(matrix, width, height, pad_size):
    """
    Description:
        - Pad the input matrix with zeros from the four direction with the specified padding size.

    Parameters:
        - matrix (numpy.ndarray): The input matrix.
        - width (int): The desired width of the padded matrix.
        - height (int): The desired height of the padded matrix.
        - pad_size (int): The size of padding to add around the matrix.

    Returns:
        - numpy.ndarray: The padded matrix.
    """
    padded_matrix = np.zeros(
        (height + 2 * pad_size, width + 2 * pad_size)
    )  # zeros matrix
    padded_matrix[pad_size : pad_size + height, pad_size : pad_size + width] = matrix
    return padded_matrix


def convolve2d_optimized(input_matrix, convolution_kernel, mode="same"):
    """
    Perform a 2D convolution of an input matrix with a convolution kernel.

    Parameters:
        - input_matrix (numpy.ndarray): The input matrix to be convolved.
        - convolution_kernel (numpy.ndarray): The kernel used for the convolution.
        - mode (str): The mode of convolution, can be 'same' (default), 'valid', or 'full'.

    Returns:
        - output_matrix (numpy.ndarray): The result of the convolution.
    """

    # Get dimensions of input matrix and kernel
    input_height, input_width = input_matrix.shape
    kernel_size = convolution_kernel.shape[0]
    padding_size = kernel_size // 2

    # Pad the input matrix
    padded_matrix = padding_matrix(
        input_matrix, input_width, input_height, pad_size=padding_size
    )

    # Create an array of offsets for convolution
    offset_array = np.arange(-padding_size, padding_size + 1)

    # Create a meshgrid of indices for convolution
    x_indices, y_indices = np.meshgrid(offset_array, offset_array, indexing="ij")

    # Add the meshgrid indices to an array of the original indices
    i_indices = (
        np.arange(padding_size, input_height + padding_size)[:, None, None]
        + x_indices.flatten()
    )
    j_indices = (
        np.arange(padding_size, input_width + padding_size)[None, :, None]
        + y_indices.flatten()
    )

    # Use advanced indexing to get the regions for convolution
    convolution_regions = padded_matrix[i_indices, j_indices].reshape(
        input_height, input_width, kernel_size, kernel_size
    )

    # Compute the convolution by multiplying the regions with the kernel and summing the results
    output_matrix = np.sum(convolution_regions * convolution_kernel, axis=(2, 3))

    return output_matrix


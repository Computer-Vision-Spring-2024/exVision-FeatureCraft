# backend.py
import os

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time
from math import cos, sin

import cv2
import matplotlib.pyplot as plt
import numpy as np

# in CMD: pip install qdarkstyle -> pip install pyqtdarktheme
import qdarktheme
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image
from PyQt5 import QtGui

# imports
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
)

# from scipy.signal import convolve2d
from scipy.signal import convolve2d
from skimage.transform import rescale, resize
from task3_ui import Ui_MainWindow


# Helper functions
def convert_to_grey(img_RGB: np.ndarray) -> np.ndarray:
    grey = np.dot(img_RGB[..., :3], [0.2989, 0.5870, 0.1140])
    return grey


def convert_BGR_to_RGB(img_BGR_nd_arr: np.ndarray) -> np.ndarray:
    img_RGB_nd_arr = img_BGR_nd_arr[..., ::-1]
    return img_RGB_nd_arr


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


class BackendClass(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ### ==== HARRIS & LAMBDA-MINUS ==== ###
        self.harris_current_image_RGB = None
        self.harris_response_operator = None
        self.eigenvalues = None
        self.change_the_icon()

        # Threshold Slider(Initially Disabled)
        self.ui.horizontalSlider_corner_tab.setEnabled(False)

        # Apply Harris Button
        self.ui.apply_harris_push_button.clicked.connect(
            lambda: self.on_apply_detectors_clicked(self.harris_current_image_RGB, 0)
        )
        self.ui.apply_harris_push_button.setEnabled(False)
        # Apply Lambda Minus
        self.ui.apply_lambda_minus_push_button.clicked.connect(
            lambda: self.on_apply_detectors_clicked(self.harris_current_image_RGB, 1)
        )
        self.ui.apply_lambda_minus_push_button.setEnabled(False)

        ### ==== SIFT ==== ###
        # Images
        self.sift_target_image = None
        self.sift_template_image = None
        self.sift_output_image = None

        # Default parameters
        self.n_octaves = 4
        self.s_value = 2
        self.sigma_base = 1.6
        self.r_ratio = 10
        self.contrast_th = 0.03
        self.confusion_factor = 0.3

        # Widgets that control the SIFT parameters
        self.ui.n_octaves_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.s_value_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.sigma_base_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.r_ratio_spin_box.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.contrast_th_slider.valueChanged.connect(self.get_new_SIFT_parameters)
        self.ui.confusion_factor_slider.valueChanged.connect(
            self.get_new_SIFT_parameters
        )
        # Apply SIFT Button
        self.ui.apply_sift.clicked.connect(self.apply_sift)
        self.ui.apply_sift.setEnabled(False)

        ### ==== General ==== ###
        # Connect menu action to load_image
        self.ui.actionLoad_Image.triggered.connect(self.load_image)

    def change_the_icon(self):
        self.setWindowIcon(QtGui.QIcon("App_Icon.png"))
        self.setWindowTitle("Computer Vision - Task 03 - Team 02")

    def load_image(self):
        # clear self.r and threshold label
        self.ui.threshold_value_label.setText("")
        self.harris_response_operator = None
        self.eigenvalues = None

        # Open file dialog if file_path is not provided
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "Images",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm)",
        )

        if file_path and isinstance(file_path, str):
            # Read the matrix, convert to rgb
            img = cv2.imread(file_path, 1)
            img = convert_BGR_to_RGB(img)

            current_tab = self.ui.tabWidget.currentIndex()

            if current_tab == 0:
                self.harris_current_image_RGB = img
                self.display_image(
                    self.harris_current_image_RGB,
                    self.ui.harris_input_figure_canvas,
                    "Input Image",
                )
                self.ui.apply_harris_push_button.setEnabled(True)
                self.ui.apply_lambda_minus_push_button.setEnabled(True)
            else:
                self.display_selection_dialog(img)
                if (
                    self.sift_target_image is not None
                    and self.sift_template_image is not None
                ):
                    self.ui.apply_sift.setEnabled(True)

            # Deactivate the slider and disconnect from apply harris function
            self.ui.horizontalSlider_corner_tab.setEnabled(False)
            try:
                self.ui.horizontalSlider_corner_tab.valueChanged.disconnect()
            except TypeError:
                pass

    def display_image(self, image, canvas, title):
        """ "
        Description:
            - Plots the given (image) in the specified (canvas)
        """
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)
        canvas.draw()

    # @staticmethod
    def display_selection_dialog(self, image):
        """
        Description:
            - Shows a message dialog box to determine whether this is the a template or the target image in SIFT

        Args:
            - image: The image to be displayed.
        """
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Select an Image")
        msgBox.setWindowTitle("Image Selection")
        msgBox.setMinimumWidth(150)

        # Set custom button text
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.button(QMessageBox.Yes).setText("Target Image")
        msgBox.button(QMessageBox.No).setText("Template")

        # Executing the message box
        response = msgBox.exec()
        if response == QMessageBox.Rejected:
            return
        else:
            if response == QMessageBox.Yes:
                self.sift_target_image = image
                self.display_image(image, self.ui.input_1_figure_canvas, "Target Image")
            elif response == QMessageBox.No:
                self.sift_template_image = image
                self.display_image(
                    image, self.ui.input_2_figure_canvas, "Template Image"
                )

    def on_apply_detectors_clicked(self, img_RGB, operator):
        if self.harris_current_image_RGB.any():
            self.ui.horizontalSlider_corner_tab.valueChanged.connect(
                lambda value: self.on_changing_threshold(value, img_RGB, operator)
            )

            if operator == 0:
                # call the function with default parameters
                start = time.time()
                self.apply_harris_detector_vectorized(img_RGB)
                finish = time.time()
                self.ui.consumed_time_label.setText(
                    f"This Operation consumed {finish-start:.3f} seconds || "
                )
                # Activate the slider and connect with apply harris function
                self.ui.horizontalSlider_corner_tab.setEnabled(True)
                self.ui.horizontalSlider_corner_tab.setMinimum(1)
                self.ui.horizontalSlider_corner_tab.setMaximum(int(10e6))
                self.ui.horizontalSlider_corner_tab.setSingleStep(10000)
                self.ui.horizontalSlider_corner_tab.setValue(10000)
                self.ui.threshold_value_label.setText(str(10000))

            elif operator == 1:
                # call the function with default parameters
                start = time.time()
                self.apply_lambda_minus_vectorized(img_RGB)
                finish = time.time()
                self.ui.consumed_time_label.setText(
                    f"This Operation consumed {finish-start:.3f} seconds || "
                )
                # Activate the slider and connect with apply lambda function
                self.ui.horizontalSlider_corner_tab.setEnabled(True)
                self.ui.horizontalSlider_corner_tab.setMinimum(1)
                self.ui.horizontalSlider_corner_tab.setMaximum(10000)
                self.ui.horizontalSlider_corner_tab.setSingleStep(1)
                self.ui.horizontalSlider_corner_tab.setValue(10)

                self.ui.threshold_value_label.setText(f"{0.01}% of max eigen value")
        return

    def on_changing_threshold(self, threshold, img_RGB, operator):
        output_img = img_RGB.copy()
        if operator == 0:
            if np.all(self.harris_response_operator != None):
                # Show the slider value using a label
                self.ui.threshold_value_label.setText(str(threshold))
                # Apply threshold and store detected corners
                corner_list = np.argwhere(self.harris_response_operator > threshold)
                # Create output image

                output_img[corner_list[:, 0], corner_list[:, 1]] = (
                    255,
                    0,
                    0,
                )  # Highlight detected corners in red
                self.display_image(
                    output_img,
                    self.ui.harris_output_figure_canvas,
                    "Harris Output Image",
                )
            elif operator == 1:
                if np.all(self.eigenvalues != None):
                    # Set the value of the threshold
                    value = self.ui.horizontalSlider_corner_tab.value() / 10000.0

                    # Show the slider value using a label
                    self.ui.threshold_value_label.setText(
                        f"{value}% of max eigen value"
                    )
                    # Apply threshold and store detected corners
                    corners = np.where(self.eigenvalues > value)

                    # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
                    for i, j in zip(*corners):
                        cv2.circle(
                            output_img, (j, i), 3, (0, 255, 0), -1
                        )  # Green color
                    self.display_image(
                        output_img,
                        self.ui.harris_output_figure_canvas,
                        "Lambda-Minus Output Image",
                    )

        return

    def apply_harris_detector_vectorized(
        self, img_RGB, window_size=5, k=0.04, threshold=10000
    ):
        if np.all(img_RGB != None):
            # Convert image to grayscale
            gray = convert_to_grey(img_RGB)

            Ix, Iy = np.gradient(gray)
            # Compute products of derivatives
            Ixx = Ix**2
            Ixy = Iy * Ix
            Iyy = Iy**2

            # Define window function
            window = np.ones((window_size, window_size))

            # Compute sums of the second moment matrix over the window
            Sxx = convolve2d_optimized(Ixx, window, mode="same")
            Sxy = convolve2d_optimized(Ixy, window, mode="same")
            Syy = convolve2d_optimized(Iyy, window, mode="same")

            # Compute determinant and trace of the second moment matrix
            det = Sxx * Syy - Sxy**2
            trace = Sxx + Syy

            # Compute corner response
            self.harris_response_operator = det - k * (trace**2)

            # Apply threshold and store detected corners
            corner_list = np.argwhere(self.harris_response_operator > threshold)
            corner_response = self.harris_response_operator[
                self.harris_response_operator > threshold
            ]

            # Create output image
            output_img = img_RGB.copy()
            output_img[corner_list[:, 0], corner_list[:, 1]] = (
                0,
                0,
                255,
            )  # Highlight detected corners in blue
            self.display_image(
                output_img, self.ui.harris_output_figure_canvas, "Harris Output Image"
            )

            return (
                list(zip(corner_list[:, 1], corner_list[:, 0], corner_response)),
                output_img,
            )

    def apply_lambda_minus_vectorized(
        self, img_RGB, window_size=5, threshold_percentage=0.01
    ):

        # Convert image to grayscale
        gray = convert_to_grey(img_RGB)
        output_image = img_RGB.copy()
        # Compute the gradient using Sobel 5x5 operator
        K_X = np.array(
            [
                [-1, -2, 0, 2, 1],
                [-2, -3, 0, 3, 2],
                [-3, -5, 0, 5, 3],
                [-2, -3, 0, 3, 2],
                [-1, -2, 0, 2, 1],
            ]
        )

        K_Y = (
            K_X.T
        )  # The kernel for vertical edges is the transpose of the kernel for horizontal edges

        gradient_x, gradient_y = convolve2d_optimized(
            gray, K_X, mode="same"
        ), convolve2d_optimized(gray, K_Y, mode="same")
        # Compute the elements of the H matrix
        H_xx = gradient_x * gradient_x
        H_yy = gradient_y * gradient_y
        H_xy = gradient_x * gradient_y
        # Compute the sum of the elements in a neighborhood (e.g., using a Gaussian kernel)
        # Define window function
        window = np.ones((5, 5))
        H_xx_sum = convolve2d_optimized(H_xx, window, mode="same") / 25
        H_yy_sum = convolve2d_optimized(H_yy, window, mode="same") / 25
        H_xy_sum = convolve2d_optimized(H_xy, window, mode="same") / 25

        # Compute the eigenvalues
        H = np.stack([H_xx_sum, H_xy_sum, H_xy_sum, H_yy_sum], axis=-1).reshape(
            -1, 2, 2
        )
        self.eigenvalues = np.linalg.eigvalsh(H).min(axis=-1).reshape(gray.shape)

        # Threshold to find corners
        threshold = threshold_percentage * self.eigenvalues.max()
        corners = np.where(self.eigenvalues > threshold)

        # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
        for i, j in zip(*corners):
            cv2.circle(output_image, (j, i), 3, (0, 255, 0), -1)  # Green color

        self.display_image(
            output_image,
            self.ui.harris_output_figure_canvas,
            "Lambda-Minus Output Image",
        )

    def clear_right_image(self):
        # Clear existing layouts before adding canvas
        for i in reversed(range(self.right_layout.count())):
            widget = self.right_layout.itemAt(i).widget()
            # Remove it from the layout list
            self.right_layout.removeWidget(widget)
            # Remove the widget from the GUI
            widget.setParent(None)

    ## ============== SIFT Methods ============== ##
    # == Setters == #
    def get_new_SIFT_parameters(self):
        self.n_octaves = self.ui.n_octaves_spin_box.value()
        self.s_value = self.ui.s_value_spin_box.value()
        self.sigma_base = self.ui.sigma_base_spin_box.value()
        self.r_ratio = self.ui.r_ratio_spin_box.value()
        self.contrast_th = self.ui.contrast_th_slider.value() / 1000
        self.confusion_factor = self.ui.confusion_factor_slider.value() / 10

        self.ui.n_octaves.setText(f"n_octaves: {self.n_octaves}")
        self.ui.s_value.setText(f"s_value: {self.s_value}")
        self.ui.sigma_base.setText(f"sigma_base: {self.sigma_base}")
        self.ui.r_ratio.setText(f"r_ratio: {self.r_ratio}")
        self.ui.contrast_th.setText(f"contrast_th: {self.contrast_th}")
        self.ui.confusion_factor.setText(f"confusion_factor: {self.confusion_factor}")

    def gaussian_filter_kernel(self, sigma, kernel_size=None):
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

    def generateGaussianKernels(self, sigma, s):
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
        gaussian_kernels.append(self.gaussian_filter_kernel(sigma))
        # generate kernel for each image in the octave
        for iterator in range(1, images_per_octave):
            # multiply the current scale level with the multiplicative factor
            scale_level *= k
            gaussian_kernels.append(self.gaussian_filter_kernel(scale_level))
        return gaussian_kernels

    def generate_octaves_pyramid(
        self, img, num_octaves=4, s_value=2, sigma=1.6, contrast_th=0.03, ratio_th=10
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
        gaussian_kernels = self.generateGaussianKernels(
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
                self.generate_gaussian_images_in_octave(
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

    def generate_gaussian_images_in_octave(
        self, image, gaussian_kernels, contrast_th, ratio_th, octave_index
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
            DOG_octave.append(
                gaussian_images_in_octave[-1] - gaussian_images_in_octave[-2]
            )
            if len(DOG_octave) > 2:
                # from each three difference of gaussians images, detect possible keypoints through extrema detection then applying keypoints localization
                # and filtering to discarde unstable keypoints
                keypoints.extend(
                    self.get_keypoints(
                        DOG_octave[-3:],
                        len(DOG_octave) - 2,
                        contrast_th,
                        ratio_th,
                        np.concatenate(
                            [o[:, :, np.newaxis] for o in DOG_octave], axis=2
                        ),
                    )
                )
        return gaussian_images_in_octave, DOG_octave, keypoints

    def get_keypoints(self, DOG_octave, k, contrast_th, ratio_th, DoG_full_array):
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

    def localize_keypoint(self, D, x, y, s):
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

    def visualize_pyramid(self, pyramid):
        fig, axes = plt.subplots(
            nrows=len(pyramid), ncols=len(pyramid[0]), figsize=(12, 12)
        )

        for i in range(len(pyramid)):
            for j in range(len(pyramid[i])):
                axes[i, j].imshow(pyramid[i][j], cmap="gray")
                axes[i, j].set_title(f"Octave {i}, Image {j}")
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_DOC_for_octave(self, DOG):
        fig, axes = plt.subplots(nrows=len(DOG), ncols=len(DOG[0]), figsize=(12, 12))

        for i in range(len(DOG)):
            for j in range(len(DOG[i])):
                axes[i, j].imshow(DOG[i][j], cmap="gray")
                axes[i, j].set_title(f"Octave {i}, Image {j}")
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.show()

    def visualize_keypoints(self, pyramid, keypoints):
        fig, axes = plt.subplots(
            nrows=len(pyramid), ncols=len(pyramid[0]), figsize=(12, 12)
        )

        for i in range(len(pyramid)):
            for j in range(len(pyramid[i])):
                axes[i, j].imshow(pyramid[i][j], cmap="gray")
                axes[i, j].set_title(f"Octave {i}, Image {j}")
                axes[i, j].axis("off")
                for kp in keypoints[i]:
                    x = kp[0]
                    y = kp[1]
                    circle = Circle((x, y), radius=2, color="r", fill=True)
                    axes[i, j].add_patch(circle)
        plt.tight_layout()
        plt.show()

    def sift_resize(self, img, ratio=None):
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
            ratio
            if ratio is not None
            else np.sqrt((1024 * 1024) / np.prod(img.shape[:2]))
        )
        newshape = list(map(lambda d: int(round(d * ratio)), img.shape[:2]))
        img = resize(img, newshape, anti_aliasing=True)
        return img, ratio


    def convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return image

    def represent_keypoints(self, keypoints, DoG):
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


    def sift_gradient(self, img):
        dx = np.array([-1, 0, 1]).reshape((1, 3))
        dy = dx.T
        gx = convolve2d(img, dx, boundary="symm", mode="same")
        gy = convolve2d(img, dy, boundary="symm", mode="same")
        magnitude = np.sqrt(gx * gx + gy * gy)
        direction = np.rad2deg(np.arctan2(gy, gx)) % 360 # to wrap the direction  
        return gx, gy, magnitude, direction

    def padded_slice(self, img, sl):
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
        output = np.zeros(output_shape, dtype=img.dtype) # padding of zeros if the indices of sl is out of the image boundaries
        output[dst[0] : dst[1], dst[2] : dst[3]] = img[src[0] : src[1], src[2] : src[3]]
        return output
    

    def dog_keypoints_orientations(self, img_gaussians, keypoints, sigma_base, num_bins=36, s=2):
        """Assigns the dominant orientation of the keypoint"""

        kps = []
        for octave_idx in range(len(img_gaussians)):  # iterate over the ocataves
            img_octave_gaussians = img_gaussians[octave_idx]
            octave_keypoints = keypoints[octave_idx]
            for idx, scale_keypoints in enumerate(octave_keypoints):
                scale_idx = idx + 1  ## This will be adjusted according to the sigma surface resulting from interpolation. (skip for now) 
                gaussian_img = img_octave_gaussians[scale_idx]
                sigma = (
                    1.5 * sigma_base * (2**octave_idx) * ((2 ** (1 / s)) ** (scale_idx)) # sigma for smoothing the magnitude accordingly (1.5 recommmended)
                )

                kernel = self.gaussian_filter_kernel(sigma)
                radius = int(round(sigma * 2))  # 2 x std == 95 %
                gx, gy, magnitude, direction = self.sift_gradient(gaussian_img)
                direction_idx = np.round(direction * num_bins / 360).astype(int) # dirction in terms of bins 

                for i, j in map(tuple, np.argwhere(scale_keypoints).tolist()):  # get the coordinates of the point
                    window = [i - radius, i + radius + 1, j - radius, j + radius + 1]  # the indices of the window to be extracted  
                    mag_win = self.padded_slice(magnitude, window)
                    dir_idx = self.padded_slice(direction_idx, window)
                    weight = mag_win * kernel  # modulate the weights according to the sigma * 1.5 (sigma at which the keypoint is detected)
                    hist = np.zeros(num_bins, dtype=np.float32)

                    for bin_idx in range(num_bins):
                        hist[bin_idx] = np.sum(weight[dir_idx == bin_idx])  # histogram is mag weighted

                    for bin_idx in np.argwhere(hist >= 0.8 * hist.max()).tolist():  #  returns list of lists 
                        angle = (bin_idx[0] + 0.5) * (360.0 / num_bins) % 360
                        kps.append((i, j, octave_idx, scale_idx, angle)) # there can be more than one descriptor to the same keypoint (another dominant angle) 
        return kps

    def rotated_subimage(self, image, center, theta, width, height):

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

    def get_gaussian_mask(self, sigma, filter_size):
        if sigma > 0:
            kernel = np.fromfunction(
                lambda x, y: (1 / (2 * np.pi * sigma**2))
                * np.exp(
                    -(
                        (x - (filter_size - 1) / 2) ** 2
                        + (y - (filter_size - 1) / 2) ** 2
                    )
                    / (2 * sigma**2)
                ),
                (filter_size, filter_size),
            )
            return kernel / np.sum(kernel)
        else:
            raise ValueError("Invalid value of Sigma")

    def extract_sift_descriptors(
        self, img_gaussians, keypoints, base_sigma, num_bins=8, s=2
    ):
        """Extract the 128 length descriptors of each keypoint besides their keypoint info (i ,j , oct_idx, scale_idx, orientation)"""

        descriptors = []
        points = []
        data = {}  #
        for i, j, oct_idx, scale_idx, orientation in keypoints:

            if "index" not in data or data["index"] != (oct_idx, scale_idx):
                data["index"] = (oct_idx, scale_idx)
                gaussian_img = img_gaussians[oct_idx][scale_idx] # must be editted in case of taylor approximation
                sigma = (
                    1.5 * base_sigma * (2**oct_idx) * ((2 ** (1 / s)) ** (scale_idx))  # scale invarance introduced to the keypoint (kernel std proportional to sigma of keypoint) 
                )
                data["kernel"] = self.get_gaussian_mask(sigma=sigma, filter_size=16) # the window size is constant 

                gx, gy, magnitude, direction = self.sift_gradient(gaussian_img)
                data["magnitude"] = magnitude
                data["direction"] = direction

            window_mag = self.rotated_subimage(
                data["magnitude"], (j, i), orientation, 16, 16
            ) # rotation to align with the domianant orientation
            window_mag = window_mag * data["kernel"]
            window_dir = self.rotated_subimage(
                data["direction"], (j, i), orientation, 16, 16
            )
            window_dir = (((window_dir - orientation) % 360) * num_bins / 360.0).astype(
                int
            ) # subtract the dominant orientation to make it direction invariance 

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
            features /= np.linalg.norm(features) # normalize 
            np.clip(features, np.finfo(np.float16).eps, 0.2, out=features)  # clip to remove non-linear illumnation effect (0.2) as descripted by autho
            features /= np.linalg.norm(features) # renormalize 
            descriptors.append(features)
            points.append((i, j, oct_idx, scale_idx, orientation))
        return points, descriptors

    def computeKeypointsAndDescriptors(
        self, image, n_octaves, s_value, sigma_base, constract_th, r_ratio
    ):
        grayscaled_image = self.convert_to_grayscale(image) # convert to grayscale 
        base_image = rescale(grayscaled_image, 2, anti_aliasing=False) # upsampling to increase the number of features extracted 
        pyramid, DoG, keypoints = self.generate_octaves_pyramid(
            base_image, n_octaves, s_value, sigma_base, constract_th, r_ratio
        )
        keypoints = self.represent_keypoints(keypoints, DoG)  # represent the keypoints in each (octave, scale) as bool images  
        keypoints_ijso = self.dog_keypoints_orientations(
            pyramid, keypoints, sigma_base, 36, s_value
        )  # ( i ,j , oct_idx, scale_idx, orientation)
        points, descriptors = self.extract_sift_descriptors(
            pyramid, keypoints_ijso, sigma_base, 8, s_value
        )
        return points, descriptors

    def kp_list_2_opencv_kp_list(self, kp_list):
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

    def match(self, img_a, pts_a, desc_a, img_b, pts_b, desc_b, tuning_distance=0.3):
        img_a, img_b = tuple(map(lambda i: np.uint8(i * 255), [img_a, img_b]))

        desc_a = np.array(desc_a, dtype=np.float32)
        desc_b = np.array(desc_b, dtype=np.float32)

        pts_a = self.kp_list_2_opencv_kp_list(pts_a)
        pts_b = self.kp_list_2_opencv_kp_list(pts_b)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_a, desc_b, k=2) # apply nearest neighbour to get the nearest 2 for each descriptor.
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < tuning_distance * n.distance: # (if evaluate to "false", then there is confusion around this descriptor, so neglect)
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

    def apply_sift(self):
        start = time.time()
        self.sift_target_image, ratio = self.sift_resize(self.sift_target_image)
        self.sift_template_image, _ = self.sift_resize(self.sift_template_image, ratio)

        img_kp, img_des = self.computeKeypointsAndDescriptors(
            self.sift_target_image,
            self.n_octaves,
            self.s_value,
            self.sigma_base,
            self.contrast_th,
            self.r_ratio,
        )
        template_kp, template_des = self.computeKeypointsAndDescriptors(
            self.sift_template_image,
            self.n_octaves,
            self.s_value,
            self.sigma_base,
            self.contrast_th,
            self.r_ratio,
        )

        img_match = self.match(
            self.sift_target_image,
            img_kp,
            img_des,
            self.sift_template_image,
            template_kp,
            template_des,
            self.confusion_factor,
        )

        self.sift_output_image = img_match
        self.display_image(img_match, self.ui.sift_output_figure_canvas, "SIFT Output")
        self.ui.tabWidget.setCurrentIndex(2)

        end = time.time()
        self.ui.sift_elapsed_time.setText(f"Elapsed Time is {end-start:.3f} seconds")
        return


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = BackendClass()
    MainWindow.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())

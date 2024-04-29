# backend.py
import os
import random
from itertools import combinations

import numpy as np

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

# from task3_ui import Ui_MainWindow
from UI import Ui_MainWindow


# Helper functions
def convert_to_grey(img_RGB: np.ndarray) -> np.ndarray:
    if len(img_RGB.shape) == 3:
        grey = np.dot(img_RGB[..., :3], [0.2989, 0.5870, 0.1140])
        return grey.astype(np.uint8)
    else:
        return img_RGB.astype(np.uint8)


def convert_BGR_to_RGB(img_BGR_nd_arr: np.ndarray) -> np.ndarray:
    img_RGB_nd_arr = img_BGR_nd_arr[..., ::-1]
    return img_RGB_nd_arr


def rgb_to_xyz(rgb):
    """Convert RGB color values to XYZ color values."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    X = 0.412453 * R + 0.35758 * G + 0.180423 * B
    Y = 0.212671 * R + 0.71516 * G + 0.072169 * B
    Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
    return np.stack((X, Y, Z), axis=-1)


def xyz_to_luv(xyz):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    constant = 903.3
    un = 0.19793943
    vn = 0.46832096

    epsilon = 1e-12  # to prevent division by zero
    u_prime = 4 * X / (X + 15 * Y + 3 * Z + epsilon)
    v_prime = 9 * Y / (X + 15 * Y + 3 * Z + epsilon)

    L = np.where(Y > 0.008856, 116 * Y ** (1 / 3) - 16, constant * Y)
    U = 13 * L * (u_prime - un)
    V = 13 * L * (v_prime - vn)

    return np.stack((L, U, V), axis=-1)


def scale_luv_8_bits(luv_image):
    L, U, V = luv_image[..., 0], luv_image[..., 1], luv_image[..., 2]

    scaled_L = L * (255 / 100)
    scaled_U = (U + 134) * (255 / 354)
    scaled_V = (V + 140) * (255 / 262)

    return np.stack((L, U, V), axis=-1)


def anti_aliasing_resize(img):
    """This function can be used for resizing images of huge size to optimize the segmentation algorithm"""
    ratio = min(1, np.sqrt((512 * 512) / np.prod(img.shape[:2])))
    newshape = list(map(lambda d: int(round(d * ratio)), img.shape[:2]))
    img = resize(img, newshape, anti_aliasing=True)
    return img


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


def gaussian_weight(distance, sigma):
    """Introduce guassian weighting based on the distance from the mean"""
    return np.exp(-(distance**2) / (2 * sigma**2))


def generate_random_color():
    """
    Description:
        -   Generate a random color for the seeds and their corresponding region in the region-growing segmentation.
    """
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def _pad_image(kernel_size, grayscale_image):
    """
    Description:
        - Pads the grayscale image with zeros.

    Returns:
        - [numpy.ndarray]: A padded grayscale image.
    """
    pad_width = kernel_size // 2
    return np.pad(
        grayscale_image,
        ((pad_width, pad_width), (pad_width, pad_width)),
        mode="edge",
    )


def Normalized_histogram_computation(Image):
    """
    Compute the normalized histogram of a grayscale image.

    Parameters:
    - Image: numpy.ndarray.

    Returns:
    - Histogram: numpy array
        A 1D array representing the normalized histogram of the input image.
        It has 256 element, each element corresponds to the probability of certain pixel intensity (0 to 255).
    """
    # Get the dimensions of the image
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    # Initialize the histogram array with zeros. The array has 256 element, each corresponding to a pixel intensity value (0 to 255)
    Histogram = np.zeros([256])

    # Compute the histogram for each pixel in each channel
    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            # Increment the count of pixels in the histogram for the same pixel intensity at position (x, y) in the image.
            # This operation updates the histogram to track the number of pixels with a specific intensity value.
            Histogram[Image[x, y]] += 1
    # Normalize the histogram by dividing each bin count by the total number of pixels in the image
    Histogram /= Image_Height * Image_Width

    return Histogram


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

        ### ==== Region-Growing ==== ###
        self.rg_input = None
        self.rg_input_grayscale = None
        self.rg_output = None
        self.rg_seeds = None
        self.rg_threshold = 20
        self.ui.region_growing_input_figure.canvas.mpl_connect(
            "button_press_event", self.rg_canvas_clicked
        )
        self.ui.region_growing_threshold_slider.valueChanged.connect(
            self.update_region_growing_threshold
        )

        # Region Growing Buttons
        self.ui.apply_region_growing.clicked.connect(self.apply_region_growing)
        self.ui.apply_region_growing.setEnabled(False)
        self.ui.reset_region_growing.clicked.connect(self.reset_region_growing)
        self.ui.reset_region_growing.setEnabled(False)

        ### ==== Agglomerative Clustering ==== ###
        self.agglo_input_image = None
        self.agglo_output_image = None
        self.agglo_number_of_clusters = 2
        self.downsampling = False
        self.agglo_scale_factor = 4
        self.ui.apply_agglomerative.setEnabled(False)
        self.ui.apply_agglomerative.clicked.connect(self.apply_agglomerative_clustering)
        self.ui.downsampling.stateChanged.connect(self.get_agglomerative_parameters)
        self.ui.agglo_scale_factor.valueChanged.connect(
            self.get_agglomerative_parameters
        )

        ### ==== K_Means ==== ###
        self.k_means_input = None
        self.k_means_luv_input = None
        self.k_means_output = None
        self.n_clusters = 4
        self.max_iterations = 4
        self.spatial_segmentation = False
        self.ui.spatial_segmentation_weight_spinbox.setEnabled(False)
        self.spatial_segmentation_weight = 1
        self.centroid_optimization = True
        self.k_means_LUV = False

        # K_Means Buttons
        self.ui.apply_k_means.setEnabled(False)
        self.ui.apply_k_means.clicked.connect(self.apply_k_means)
        self.ui.spatial_segmentation.stateChanged.connect(
            self.enable_spatial_segmentation
        )

        ### ==== Mean-Shift ==== ###
        self.mean_shift_input = None
        self.mean_shift_luv_input = None
        self.mean_shift_output = None
        self.mean_shift_window_size = 200
        self.mean_shift_sigma = 20
        self.mean_shift_threshold = 10
        self.mean_shift_luv = False

        # Mean-Shift Buttons
        self.ui.apply_mean_shift.setEnabled(False)
        self.ui.apply_mean_shift.clicked.connect(self.apply_mean_shift)

        ### ==== Thresholding ==== ###
        self.thresholding_grey_input = None
        self.thresholding_output = None
        self.number_of_thresholds = 2
        self.thresholding_type = "Optimal - Binary"
        self.local_or_global = "Global"
        self.otsu_step = 1
        self.separability_measure = 0
        self.global_thresholds = None
        self.ui.thresholding_comboBox.currentIndexChanged.connect(
            self.get_thresholding_parameters
        )

        # Thresholding Buttons and checkbox
        self.ui.apply_thresholding.setEnabled(False)
        self.ui.apply_thresholding.clicked.connect(self.apply_thresholding)
        self.ui.number_of_thresholds_slider.setEnabled(False)
        self.ui.number_of_thresholds_slider.valueChanged.connect(
            self.get_thresholding_parameters
        )
        self.ui.local_checkbox.stateChanged.connect(self.local_global_thresholding)
        self.ui.global_checkbox.stateChanged.connect(self.local_global_thresholding)
        self.ui.otsu_step_spinbox.setEnabled(False)

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
                    False,
                )
                self.ui.apply_harris_push_button.setEnabled(True)
                self.ui.apply_lambda_minus_push_button.setEnabled(True)
            elif current_tab == 1:
                self.display_selection_dialog(img)
                if (
                    self.sift_target_image is not None
                    and self.sift_template_image is not None
                ):
                    self.ui.apply_sift.setEnabled(True)
            elif current_tab == 3:
                self.rg_input = img
                self.rg_input_grayscale = convert_to_grey(self.rg_input)
                self.rg_output = img
                self.display_image(
                    self.rg_input,
                    self.ui.region_growing_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.display_image(
                    self.rg_output,
                    self.ui.region_growing_output_figure_canvas,
                    "Output Image",
                    False,
                )
                self.ui.apply_region_growing.setEnabled(True)
                self.ui.reset_region_growing.setEnabled(True)
            elif current_tab == 4:
                self.agglo_input_image = img
                self.display_image(
                    self.agglo_input_image,
                    self.ui.agglomerative_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.ui.apply_agglomerative.setEnabled(True)
            elif current_tab == 5:
                self.k_means_luv_input = self.map_rgb_luv(img)
                self.k_means_input = img

                if self.ui.k_means_LUV_conversion.isChecked():
                    self.display_image(
                        self.k_means_luv_input,
                        self.ui.k_means_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                else:
                    self.display_image(
                        self.k_means_input,
                        self.ui.k_means_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                self.ui.apply_k_means.setEnabled(True)
            elif current_tab == 6:
                self.mean_shift_luv_input = self.map_rgb_luv(img)
                self.mean_shift_input = img

                if self.ui.mean_shift_LUV_conversion.isChecked():
                    self.display_image(
                        self.mean_shift_luv_input,
                        self.ui.mean_shift_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                else:
                    self.display_image(
                        self.mean_shift_input,
                        self.ui.mean_shift_input_figure_canvas,
                        "Input Image",
                        False,
                    )
                self.ui.apply_mean_shift.setEnabled(True)
            elif current_tab == 7:
                self.thresholding_grey_input = convert_to_grey(img)
                self.ui.number_of_thresholds_slider.setEnabled(True)
                self.display_image(
                    self.thresholding_grey_input,
                    self.ui.thresholding_input_figure_canvas,
                    "Input Image",
                    True,
                )
                self.ui.apply_thresholding.setEnabled(True)

            # Deactivate the slider and disconnect from apply harris function
            self.ui.horizontalSlider_corner_tab.setEnabled(False)
            try:
                self.ui.horizontalSlider_corner_tab.valueChanged.disconnect()
            except TypeError:
                pass

    def display_image(
        self, image, canvas, title, grey, hist_or_not=False, axis_disabled="off"
    ):
        """ "
        Description:
            - Plots the given (image) in the specified (canvas)
        """
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        if not hist_or_not:
            if not grey:
                ax.imshow(image)
            elif grey:
                ax.imshow(image, cmap="gray")
        else:
            if grey:
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
            else:
                image = convert_to_grey(image)
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")

        ax.axis(axis_disabled)
        ax.set_title(title)
        canvas.figure.subplots_adjust(left=0.20, right=0.80, bottom=0.20, top=0.80)
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
                self.display_image(
                    image,
                    self.ui.input_1_figure_canvas,
                    "Target Image",
                    False,
                )
            elif response == QMessageBox.No:
                self.sift_template_image = image
                self.display_image(
                    image,
                    self.ui.input_2_figure_canvas,
                    "Template Image",
                    False,
                )

    ## ================ Conver to LUV colorspace ================ ##
    def map_rgb_luv(self, image):
        image = anti_aliasing_resize(image)
        normalized_image = (image - image.min()) / (
            image.max() - image.min()
        )  # nomalize before
        xyz_image = rgb_to_xyz(normalized_image)
        luv_image = xyz_to_luv(xyz_image)
        luv_image_normalized = (luv_image - luv_image.min()) / (
            luv_image.max() - luv_image.min()
        )  # normalize after  (point of question !!)
        # scaled_image = scale_luv_8_bits(luv_image)
        return luv_image_normalized

    ## ============== Harris & Lambda-Minus Methods ============== ##
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
                    False,
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
                        False,
                    )

        return

    def apply_harris_detector_vectorized(
        self, img_RGB, window_size=5, k=0.04, threshold=10000
    ):
        """
        Apply the Harris corner detection algorithm on an RGB image in a vectorized manner.

        This method detects corners within an image using the Harris corner detection algorithm. It converts the image to grayscale, computes the gradients, and then calculates the second moment matrix. The corner response is determined by the determinant and trace of this matrix, and corners are identified based on a specified threshold.

        Parameters:
        - img_RGB (numpy.ndarray): The input image in RGB format.
        - window_size (int, optional): The size of the window used to compute the sums of the second moment matrix. Defaults to 5.
        - k (float, optional): The sensitivity factor to separate corners from edges, typically between 0.04-0.06. Defaults to 0.04.
        - threshold (int, optional): The threshold above which a response is considered a corner. Defaults to 10000.

        Returns:
        - A tuple containing:
            - A list of tuples with the x-coordinate, y-coordinate, and corner response value for each detected corner.
            - The output image with detected corners highlighted in blue.

        The method modifies the input image by highlighting detected corners in blue and displays the result using the `display_image` method.
        """
        if np.all(img_RGB != None):
            # Convert image to grayscale
            gray = convert_to_grey(img_RGB)
            self.display_image(
                gray,
                self.ui.harris_input_figure_canvas,
                "Input Image",
                False,
            )
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
                output_img,
                self.ui.harris_output_figure_canvas,
                "Harris Output Image",
                False,
            )

            return (
                list(zip(corner_list[:, 1], corner_list[:, 0], corner_response)),
                output_img,
            )

    def apply_lambda_minus_vectorized(
        self, img_RGB, window_size=5, threshold_percentage=0.01
    ):
        """
        Apply the Lambda-Minus corner detection algorithm on an RGB image.

        This method implements a vectorized approach to identify corners within an image using the Lambda-Minus algorithm. It involves converting the image to grayscale, computing gradients, constructing the Hessian matrix, and finding eigenvalues to determine corner points based on a specified threshold.

        Parameters:
        - img_RGB (numpy.ndarray): The input image in RGB format.
        - window_size (int, optional): The size of the window used to compute the sum of Hessian matrix elements. Defaults to 5.
        - threshold_percentage (float, optional): The percentage of the maximum eigenvalue used to set the threshold for corner detection. Defaults to 0.01.

        Returns:
        - output_image (numpy.ndarray): The RGB image with detected corners marked in green.

        The method modifies the input image by drawing green circles at the detected corner points and displays the result using the `display_image` method.
        """

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
            False,
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
        direction = np.rad2deg(np.arctan2(gy, gx)) % 360  # to wrap the direction
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
        output = np.zeros(
            output_shape, dtype=img.dtype
        )  # padding of zeros if the indices of sl is out of the image boundaries
        output[dst[0] : dst[1], dst[2] : dst[3]] = img[src[0] : src[1], src[2] : src[3]]
        return output

    def dog_keypoints_orientations(
        self, img_gaussians, keypoints, sigma_base, num_bins=36, s=2
    ):
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

                kernel = self.gaussian_filter_kernel(sigma)
                radius = int(round(sigma * 2))  # 2 x std == 95 %
                gx, gy, magnitude, direction = self.sift_gradient(gaussian_img)
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
                    mag_win = self.padded_slice(magnitude, window)
                    dir_idx = self.padded_slice(direction_idx, window)
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
                data["kernel"] = self.get_gaussian_mask(
                    sigma=sigma, filter_size=16
                )  # the window size is constant

                gx, gy, magnitude, direction = self.sift_gradient(gaussian_img)
                data["magnitude"] = magnitude
                data["direction"] = direction

            window_mag = self.rotated_subimage(
                data["magnitude"], (j, i), orientation, 16, 16
            )  # rotation to align with the domianant orientation
            window_mag = window_mag * data["kernel"]
            window_dir = self.rotated_subimage(
                data["direction"], (j, i), orientation, 16, 16
            )
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
        self, image, n_octaves, s_value, sigma_base, constract_th, r_ratio
    ):
        grayscaled_image = self.convert_to_grayscale(image)  # convert to grayscale
        base_image = rescale(
            grayscaled_image, 2, anti_aliasing=False
        )  # upsampling to increase the number of features extracted
        pyramid, DoG, keypoints = self.generate_octaves_pyramid(
            base_image, n_octaves, s_value, sigma_base, constract_th, r_ratio
        )
        keypoints = self.represent_keypoints(
            keypoints, DoG
        )  # represent the keypoints in each (octave, scale) as bool images
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
        self.display_image(
            img_match,
            self.ui.sift_output_figure_canvas,
            "SIFT Output",
            False,
        )
        self.ui.tabWidget.setCurrentIndex(2)

        end = time.time()
        self.ui.sift_elapsed_time.setText(f"Elapsed Time is {end-start:.3f} seconds")
        return

    ## ============== Region-Growing Methods ============== ##
    def rg_canvas_clicked(self, event):
        if event.xdata is not None and event.ydata is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            print(
                f"Clicked pixel at ({x}, {y}) with value {self.rg_input_grayscale[y, x]}"
            )

            # Plot a dot at the clicked location
            ax = self.ui.region_growing_input_figure_canvas.figure.gca()
            ax.scatter(
                x, y, color="red", s=10
            )  # Customize the color and size as needed
            self.ui.region_growing_input_figure_canvas.draw()

            # Store the clicked coordinates as seeds
            if self.rg_seeds is None:
                self.rg_seeds = [(x, y)]
            else:
                self.rg_seeds.append((x, y))

    def update_region_growing_threshold(self):
        self.rg_threshold = self.ui.region_growing_threshold_slider.value()
        self.ui.region_growing_threshold.setText(f"Threshold: {self.rg_threshold}")

    def apply_region_growing(self):
        """
        Perform region growing segmentation.

        Parameters:
            image (numpy.ndarray): Input image.
            seeds (list): List of seed points (x, y).
            threshold (float): Threshold for similarity measure.

        Returns:
            numpy.ndarray: Segmented image.
        """
        # Initialize visited mask and segmented image

        # 'visited' is initialized to keep track of which pixels have been visited (Mask)
        visited = np.zeros_like(self.rg_input_grayscale, dtype=bool)
        # 'segmented' will store the segmented image where each pixel belonging
        # to a region will be marked with the corresponding color
        segmented = np.zeros_like(self.rg_input)

        # Define 3x3 window for mean calculation
        window_size = 3
        half_window = window_size // 2

        # Loop through seed points
        for seed in self.rg_seeds:
            seed_x, seed_y = seed

            # Check if seed coordinates are within image bounds
            if (
                0 <= seed_x < self.rg_input_grayscale.shape[0]
                and 0 <= seed_y < self.rg_input_grayscale.shape[1]
            ):
                # Process the seed point
                region_mean = self.rg_input_grayscale[seed_x, seed_y]

            # Initialize region queue with seed point
            # It holds the candidate pixels
            queue = [(seed_x, seed_y)]

            # Region growing loop
            # - Breadth-First Search (BFS) is used here to ensure
            # that all similar pixels are added to the region
            while queue:
                # Pop pixel from queue
                x, y = queue.pop(0)

                # Check if pixel is within image bounds and not visited
                if (
                    (0 <= x < self.rg_input_grayscale.shape[0])
                    and (0 <= y < self.rg_input_grayscale.shape[1])
                    and not visited[x, y]
                ):
                    # Mark pixel as visited
                    visited[x, y] = True

                    # Check similarity with region mean
                    if (
                        abs(self.rg_input_grayscale[x, y] - region_mean)
                        <= self.rg_threshold
                    ):
                        # Add pixel to region
                        segmented[x, y] = self.rg_input[x, y]

                        # Update region mean
                        region_pixels = segmented[segmented != 0]
                        region_mean = np.mean(region_pixels)

                        # Update region mean
                        # Incremental update formula for mean:
                        # new_mean = (old_mean * n + new_value) / (n + 1)
                        # n = np.sum(segmented != 0)  # Number of pixels in the region
                        # region_mean = (
                        #     region_mean * n + self.rg_input_grayscale[x, y]
                        # ) / (n + 1)

                        # Add neighbors to queue
                        for i in range(-half_window, half_window + 1):
                            for j in range(-half_window, half_window + 1):
                                if (
                                    0 <= x + i < self.rg_input_grayscale.shape[0]
                                    and 0 <= y + j < self.rg_input_grayscale.shape[1]
                                ):
                                    queue.append((x + i, y + j))

        self.plot_rg_output(segmented)
        # self.display_image(segmented, self.ui.sift_output_figure_canvas, "SIFT Output")

    def plot_rg_output(self, segmented_image):
        ## =========== Display the segmented image =========== ##
        # Find contours of segmented region
        contours, _ = cv2.findContours(
            cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Draw contours on input image
        output_image = self.rg_input.copy()
        cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

        # Display the output image
        self.display_image(
            output_image,
            self.ui.region_growing_output_figure_canvas,
            "Region Growing Output",
            False,
        )

    def reset_region_growing(self):
        self.rg_seeds = None
        self.rg_threshold = 20
        self.ui.region_growing_threshold_slider.setValue(self.rg_threshold)
        self.ui.region_growing_threshold.setText(f"Threshold: {self.rg_threshold}")
        self.rg_output = self.rg_input
        self.display_image(
            self.rg_input,
            self.ui.region_growing_input_figure_canvas,
            "Input Image",
            False,
        )
        self.display_image(
            self.rg_output,
            self.ui.region_growing_output_figure_canvas,
            "Region Growing Output",
            False,
        )

    ## ============== K-Means Methods ============== ##
    def get_new_k_means_parameters(self):
        self.n_clusters = self.ui.n_clusters_spinBox.value()
        self.max_iterations = self.ui.k_means_max_iteratation_spinBox.value()
        self.centroid_optimization = self.ui.centroid_optimization.isChecked()

        self.spatial_segmentation_weight = (
            self.ui.spatial_segmentation_weight_spinbox.value()
        )

        self.spatial_segmentation = self.ui.spatial_segmentation.isChecked()
        self.k_means_LUV = self.ui.k_means_LUV_conversion.isChecked()

    def enable_spatial_segmentation(self):
        if self.ui.spatial_segmentation.isChecked():
            self.spatial_segmentation = True
            self.ui.spatial_segmentation_weight_spinbox.setEnabled(True)
        else:
            self.spatial_segmentation = False
            self.ui.spatial_segmentation_weight_spinbox.setEnabled(False)

    def kmeans_segmentation(
        self,
        image,
        max_iterations,
        centroids_color=None,
        centroids_spatial=None,
    ):
        """
        Perform K-means clustering segmentation on an input image.

        Parameters:
        - centroids_color (numpy.ndarray, optional): Initial centroids in terms of color. Default is None.
        - centroids_spatial (numpy.ndarray, optional): Initial centroids in terms of spatial coordinates. Default is None.

        Returns:
        If include_spatial_seg is False:
        - centroids_color (numpy.ndarray): Final centroids in terms of color.
        - labels (numpy.ndarray): Labels of each pixel indicating which cluster it belongs to.

        If include_spatial_seg is True:
        - centroids_color (numpy.ndarray): Final centroids in terms of color.
        - centroids_spatial (numpy.ndarray): Final centroids in terms of spatial coordinates.
        - labels (numpy.ndarray): Labels of each pixel indicating which cluster it belongs to.
        """
        img = np.array(image, copy=True, dtype=float)

        if self.spatial_segmentation:
            h, w, _ = img.shape
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            xy_coords = np.column_stack(
                (x_coords.flatten(), y_coords.flatten())
            )  # spatial coordinates in the features space

        img_as_features = img.reshape(-1, img.shape[2])  # without spatial info included

        labels = np.zeros(
            (img_as_features.shape[0], 1)
        )  # (image size x 1) this array contains the labels of each pixel (belongs to which centroid)

        distance = np.zeros(
            (img_as_features.shape[0], self.n_clusters), dtype=float
        )  # (distance for each colored pixel over the entire clusters)

        # if the centriods have been not provided
        if centroids_color is None:
            centroids_indices = np.random.choice(
                img_as_features.shape[0], self.n_clusters, replace=False
            )  # initialize the centroids
            centroids_color = img_as_features[centroids_indices]  # in terms of color
            if self.spatial_segmentation:
                centroids_spatial = xy_coords[
                    centroids_indices
                ]  # this to introduce restriction in the spatial space of the image

            # Form initial clustering
            if self.centroid_optimization:
                rows = np.arange(img.shape[0])
                columns = np.arange(img.shape[1])

                sample_size = (
                    len(rows) // 16 if len(rows) > len(columns) else len(columns) // 16
                )
                ii = np.random.choice(rows, size=sample_size, replace=False)
                jj = np.random.choice(columns, size=sample_size, replace=False)
                subimage = img[
                    ii[:, np.newaxis], jj[np.newaxis, :], :
                ]  # subimage for redistribute the centriods

                if self.spatial_segmentation:
                    centroids_color, centroids_spatial, _ = self.kmeans_segmentation(
                        subimage,
                        max_iterations // 2,
                        centroids_color=centroids_color,
                        centroids_spatial=centroids_spatial,
                    )
                else:
                    centroids_color, _ = self.kmeans_segmentation(
                        subimage,
                        max_iterations // 2,
                        centroids_color=centroids_color,
                    )

        for _ in range(max_iterations):
            for centroid_idx in range(centroids_color.shape[0]):
                distance[:, centroid_idx] = np.linalg.norm(
                    img_as_features - centroids_color[centroid_idx], axis=1
                )

                if self.spatial_segmentation:
                    distance[:, centroid_idx] += (
                        np.linalg.norm(
                            xy_coords - centroids_spatial[centroid_idx], axis=1
                        )
                        * self.spatial_segmentation_weight
                    )

            labels = np.argmin(
                distance, axis=1
            )  # assign each point in the feature space a label according to its distance from each centriod based on (spatial and color distance)

            for centroid_idx in range(centroids_color.shape[0]):
                cluster_colors = img_as_features[labels == centroid_idx]
                if len(cluster_colors) > 0:  # Check if cluster is not empty
                    new_centroid_color = np.mean(cluster_colors, axis=0)
                    centroids_color[centroid_idx] = new_centroid_color

                    if self.spatial_segmentation:
                        cluster_spatial = xy_coords[labels == centroid_idx]
                        new_centroid_spatial = np.mean(cluster_spatial, axis=0)
                        centroids_spatial[centroid_idx] = new_centroid_spatial

        if self.spatial_segmentation:
            return centroids_color, centroids_spatial, labels
        else:
            return centroids_color, labels

    def apply_k_means(self):
        self.get_new_k_means_parameters()
        if self.spatial_segmentation:
            if self.k_means_LUV:
                self.display_image(
                    self.k_means_luv_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, _, labels = self.kmeans_segmentation(
                    self.k_means_luv_input, self.max_iterations
                )
            else:
                self.display_image(
                    self.k_means_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, _, labels = self.kmeans_segmentation(
                    self.k_means_input, self.max_iterations
                )

        else:
            if self.k_means_LUV:
                self.display_image(
                    self.k_means_luv_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, labels = self.kmeans_segmentation(
                    self.k_means_luv_input, self.max_iterations
                )
            else:
                self.display_image(
                    self.k_means_input,
                    self.ui.k_means_input_figure_canvas,
                    "Input Image",
                    False,
                )
                centroids_color, labels = self.kmeans_segmentation(
                    self.k_means_input, self.max_iterations
                )

        self.k_means_output = centroids_color[labels]

        if self.k_means_LUV:
            self.k_means_output = self.k_means_output.reshape(
                self.k_means_luv_input.shape
            )
        else:
            self.k_means_output = self.k_means_output.reshape(self.k_means_input.shape)

        self.k_means_output = (self.k_means_output - self.k_means_output.min()) / (
            self.k_means_output.max() - self.k_means_output.min()
        )
        self.display_image(
            self.k_means_output,
            self.ui.k_means_output_figure_canvas,
            "K-Means Output",
            False,
        )

    ## ============== Mean-Shift Methods ============== ##
    def get_new_mean_shift_parameters(self):
        self.mean_shift_window_size = self.ui.mean_shift_window_size_spinbox.value()
        self.mean_shift_sigma = self.ui.mean_shift_sigma_spinbox.value()
        self.mean_shift_threshold = self.ui.mean_shift_threshold_spinbox.value()

        self.mean_shift_luv = self.ui.mean_shift_LUV_conversion.isChecked()

    def mean_shift_clusters(self, image, window_size, threshold, sigma):
        """
        Perform Mean Shift clustering on an image.

        Args:
            image (numpy.ndarray): The input image.
            window_size (float): The size of the window for the mean shift.
            threshold (float): The convergence threshold.
            sigma (float): The standard deviation for the Gaussian weighting.

        Returns:
            list: A list of dictionaries representing the clusters. Each dictionary contains:
                - 'points': A boolean array indicating the points belonging to the cluster.
                - 'center': The centroid of the cluster.
        """
        image = (
            (image - image.min()) * (1 / (image.max() - image.min())) * 255
        ).astype(np.uint8)
        img = np.array(image, copy=True, dtype=float)

        img_as_features = img.reshape(
            -1, img.shape[2]
        )  # feature space (each channel elongated)

        num_points = len(img_as_features)
        visited = np.full(num_points, False, dtype=bool)
        clusters = []

        while (
            np.sum(visited) < num_points
        ):  # check if all points have been visited, thus, assigned a cluster.
            initial_mean_idx = np.random.choice(
                np.arange(num_points)[np.logical_not(visited)]
            )
            initial_mean = img_as_features[initial_mean_idx]

            while True:
                distances = np.linalg.norm(
                    initial_mean - img_as_features, axis=1
                )  # distances

                weights = gaussian_weight(
                    distances, sigma
                )  # weights for computing new mean

                within_window = np.where(distances <= window_size / 2)[0]
                within_window_bool = np.full(num_points, False, dtype=bool)
                within_window_bool[within_window] = True

                within_window_points = img_as_features[within_window]

                new_mean = np.average(
                    within_window_points, axis=0, weights=weights[within_window]
                )

                # Check convergence
                if np.linalg.norm(new_mean - initial_mean) < threshold:
                    merged = False  # Check merge condition
                    for cluster in clusters:
                        if (
                            np.linalg.norm(cluster["center"] - new_mean)
                            < 0.5 * window_size
                        ):
                            # Merge with existing cluster
                            cluster["points"] = (
                                cluster["points"] + within_window_bool
                            )  # bool array that represent the points of each cluster
                            cluster["center"] = 0.5 * (cluster["center"] + new_mean)
                            merged = True
                            break

                    if not merged:
                        # No merge, create new cluster
                        clusters.append(
                            {"points": within_window_bool, "center": new_mean}
                        )

                    visited[within_window] = True
                    break

                initial_mean = new_mean

        return clusters

    def calculate_mean_shift_clusters(self, image):
        clusters = self.mean_shift_clusters(
            image,
            self.mean_shift_window_size,
            self.mean_shift_threshold,
            self.mean_shift_sigma,
        )
        output = np.zeros(image.shape)

        for cluster in clusters:
            bool_image = cluster["points"].reshape(image.shape[0], image.shape[1])
            output[bool_image, :] = cluster["center"]

        return output

    def apply_mean_shift(self):
        self.get_new_mean_shift_parameters()

        if self.mean_shift_luv:
            self.display_image(
                self.mean_shift_luv_input,
                self.ui.mean_shift_input_figure_canvas,
                "Input Image",
                False,
            )
            self.mean_shift_output = self.calculate_mean_shift_clusters(
                self.mean_shift_luv_input
            )
        else:
            self.display_image(
                self.mean_shift_input,
                self.ui.mean_shift_input_figure_canvas,
                "Input Image",
                False,
            )
            self.mean_shift_output = self.calculate_mean_shift_clusters(
                self.mean_shift_input
            )

        self.mean_shift_output = (
            self.mean_shift_output - self.mean_shift_output.min()
        ) / (self.mean_shift_output.max() - self.mean_shift_output.min())
        self.display_image(
            self.mean_shift_output,
            self.ui.mean_shift_output_figure_canvas,
            "Mean Shift Output",
            False,
        )

    ## ============== Thresholding Methods ============== ##
    def get_thresholding_parameters(self):
        self.number_of_thresholds = self.ui.number_of_thresholds_slider.value()
        self.thresholding_type = self.ui.thresholding_comboBox.currentText()
        self.otsu_step = self.ui.otsu_step_spinbox.value()
        self.ui.number_of_thresholds.setText(
            "Number of thresholds: " + str(self.number_of_thresholds)
        )
        if self.thresholding_type == "OTSU":
            self.ui.number_of_thresholds_slider.setEnabled(True)
            self.ui.otsu_step_spinbox.setEnabled(True)
        else:
            self.ui.number_of_thresholds_slider.setEnabled(False)
            self.ui.otsu_step_spinbox.setEnabled(False)

    def local_global_thresholding(self, state):
        sender = self.sender()
        if state == 2:  # Checked state
            if sender == self.ui.local_checkbox:
                self.ui.global_checkbox.setChecked(False)
                self.local_or_global = "Local"
            else:
                self.ui.local_checkbox.setChecked(False)
                self.local_or_global = "Global"

    def apply_thresholding(self):
        self.get_thresholding_parameters()
        if self.thresholding_type == "Optimal - Binary":
            if self.local_or_global == "Local":
                self.thresholding_output = self.local_thresholding(
                    self.thresholding_grey_input, self.optimal_thresholding
                )
            elif self.local_or_global == "Global":
                self.thresholding_output, self.global_thresholds, _ = (
                    self.optimal_thresholding(self.thresholding_grey_input)
                )
                self.display_image(
                    self.thresholding_grey_input,
                    self.ui.histogram_global_thresholds_figure_canvas,
                    "Histogram",
                    True,
                    True,
                    "on",
                )

        elif self.thresholding_type == "OTSU":
            if self.local_or_global == "Local":
                self.thresholding_output = self.local_thresholding(
                    grayscale_image=self.thresholding_grey_input,
                    threshold_algorithm=lambda img: self.multi_otsu(
                        img, self.number_of_thresholds, self.otsu_step
                    ),
                    kernel_size=5,
                )
            elif self.local_or_global == "Global":
                (
                    self.thresholding_output,
                    self.global_thresholds,
                    self.separability_measure,
                ) = self.multi_otsu(
                    self.thresholding_grey_input,
                    self.number_of_thresholds,
                    self.otsu_step,
                )
                self.display_image(
                    self.thresholding_grey_input,
                    self.ui.histogram_global_thresholds_figure_canvas,
                    "Histogram",
                    True,
                    True,
                    "on",
                )
                self.ui.separability_measure.setText(
                    "Separability Measure = {:.3f}".format(self.separability_measure)
                )

        self.display_image(
            self.thresholding_output,
            self.ui.thresholding_output_figure_canvas,
            "Thresholding Output",
            True,
        )

    def optimal_thresholding(self, image):
        """
        Description:
            - Applies optimal thresholding to an image.

        Args:
            - image: the image to be thresholded

        Returns:
            - [numpy ndarray]: the resulted thresholded image after applying optimal threshoding algorithm.
        """
        # Initially the four corner pixels are considered the background and the rest of the pixels are the object
        corners = [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]
        # Calculate the mean of the background class
        background_mean = np.sum(corners) / 4
        # Calculate the mean of the object class by summing the intensities of the image then subtracting the four corners then dividing by the number
        # of pixels in the full image - 4
        object_mean = (np.sum(image) - np.sum(corners)) / (
            image.shape[0] * image.shape[1] - 4
        )
        # Set random iinitial values for the thresholds
        threshold = -1
        prev_threshold = 0
        # keep updating the threshold based on the means of the two classes until the new threshold equals the previous one
        while (abs(threshold - prev_threshold)) > 0:
            # Store the threshold value before updating it to compare it to the new one in the next iteration
            prev_threshold = threshold
            # Compute the new threshold value midway between the two means of the two classes
            threshold = (background_mean + object_mean) / 2
            # Get the indices whose intensity values are less than the threshold
            background_pixels = np.where(image < threshold)
            # Get the indices whose intensity values are more than the threshold
            object_pixels = np.where(image > threshold)
            if not len(background_pixels[0]) == 0:
                # Compute the new mean of the background class based on the new threshold
                background_mean = np.sum(image[background_pixels]) / len(
                    background_pixels[0]
                )
            if not len(object_pixels[0]) == 0:
                # Compute the new mean of the object class based on the new threshold
                object_mean = np.sum(image[object_pixels]) / len(object_pixels[0])
        # Set background pixels white
        image[background_pixels] = 0
        # Set object pixels black
        image[object_pixels] = 255
        return image, [[threshold]], threshold

    def otsu_thresholding(self, image):
        """
        Description:
            - Applies Otsu thresholding to an image.

        Args:
            - image: the image to be thresholded

        Returns:
            - [numpy ndarray]: the resulted thresholded image after applying Otsu threshoding algorithm.
        """
        # Calculate the normalized histogram of the image
        normalized_histogram = Normalized_histogram_computation(image)
        # initialize the weighted sum to be equal 0, which corrisponds to the weighted sum of the zero intensity (0* P(0))
        weighted_sum = 0
        # initialize the probability of class one to be equal to the probability of the 0 intensity
        probability_class1 = normalized_histogram[0]
        # Calculate the mean of the image
        global_mean = np.sum(
            np.arange(len(normalized_histogram)) * normalized_histogram
        )
        # Calculate the variance of the image
        global_variance = np.sum(
            ((np.arange(len(normalized_histogram)) - global_mean) ** 2)
            * normalized_histogram
        )
        # Variable to track the maximum between_class_variance achieved through different thresholds
        maximum_variance = 0
        # Array to store the thresholds at which the between_class_variance has a maximum value
        threshold_values = []
        # Loop over all intensity levels and try them as thresholds, then compute the between_class_variance to check the separability measure according to this threshold value
        for k in range(1, 256):
            # The probability of class1 is calculated through the cumulative sum of the probabilities of all the intensities smaller than or equal the threshold
            probability_class1 += normalized_histogram[k]
            weighted_sum += k * normalized_histogram[k]
            # if probability of class 1 equals zero or one, this means that according to the current threshold there is a single class, then there is no between_class_variance
            if probability_class1 * (1 - probability_class1) == 0:
                continue
            # This form for calculating the between_class_variance is obtained from substituting with those two equations: P1+P2=1, P1 *m1 +P2 * m2= mg
            # in the ordinary form for calculating the between_class_variance ( between_class_variance= P1*P2 (m1-m2)**2)
            # this form is slightly more efficient computationally than the ordinary form
            # because the global mean, mG, is computed only once, so only two parameters, weighted_sum and probability_class1, need to be computed for any value of k.
            between_class_variance = (
                ((global_mean * probability_class1) - weighted_sum) ** 2
            ) / (probability_class1 * (1 - probability_class1))
            if between_class_variance > maximum_variance:
                maximum_variance = between_class_variance
                # If the between_class_variance corrisponding to this threshold intensity is maximum, store the threshold value
                threshold_values = [k]
                # after connecting the backend with the UI this is recommended to be attribute to self, in order to display its value in a line edit in the UI
                separability_measure = between_class_variance / global_variance
                # To handel the case when there is more than one threshold value, maximize the between_class_variance, the optimal threshold in this case is their avg
            elif between_class_variance == maximum_variance:
                threshold_values.append(k)
        if len(threshold_values) > 1:
            # Get the average of the thresholds that maximize the between_class_variance
            threshold = np.mean(threshold_values)
        elif len(threshold_values) == 1:
            # if single threshold maximize the between_class_variance, then this is the perfect threshold to separate the classes
            threshold = threshold[0]
        else:
            # If no maximum between_class_variance then all the pixels belong to the same class (single intensity level), so assign them all to background class
            # we do so for simplicity since this often happen in the background pixels in the local thresholding, it rarely happen that the whole window has single
            # intensity in the object pixels
            image[np.where(image > 0)] = 255
            return image
        background_pixels = np.where(image < threshold)
        object_pixels = np.where(image > threshold)
        image[background_pixels] = 0
        image[object_pixels] = 255
        return image

    def local_thresholding(self, grayscale_image, threshold_algorithm, kernel_size=5):
        """
        Description:
            - Applies local thresholding to an image.

        Args:
            - grayscale_image: the image to be thresholded
            - threshold_algorithm: the algorithm through which local thresholding will be applied.
            - kernel_size: the size of the window used in local thresholding

        Returns:
            - [numpy ndarray]: the resulted thresholded image after applying the selected threshoding algorithm.
        """
        # Pad the image to avoid lossing information of the boundry pixels or getting out of bounds
        padded_image = _pad_image(kernel_size, grayscale_image)
        thresholded_image = np.zeros_like(grayscale_image)
        for i in range(0, grayscale_image.shape[0], 4):
            for j in range(0, grayscale_image.shape[1], 4):
                # Take the current pixel and its neighboors to apply the thresholding algorithm on them
                window = padded_image[i : i + kernel_size, j : j + kernel_size]
                # If all the pixels belong to the same class (single intensity level), assign them all to background class
                # we do so for simplicity since this often happen in the background pixels in the local thresholding, it rarely happen that the whole window has single
                # intensity in the object pixels
                if np.all(window == window[0, 0]):
                    thresholded_image[i, j] = 255
                    thresholded_window = window
                else:
                    # Assign the value of the middle pixel of the thresholded window to the current pixel of the thresholded image
                    thresholded_window, _, _ = threshold_algorithm(window)
                    thresholded_image[
                        i : i + kernel_size // 2, j : j + kernel_size // 2
                    ] = thresholded_window[: kernel_size // 2, : kernel_size // 2]
        return thresholded_image

    def generate_combinations(self, k, step, start=1, end=255):
        """
        Generate proper combinations of thresholds for histogram bins based on the number of thresholds

        Parameters:
        - k: the number of thresholds
        - step: the increment distance for the threshold, not 1 by default for optimization purposes
        - start: the first number in histogram bins which is 1 by default.
        - end: last number in histogram bins which is 255 by default.

        Returns:
        a list of the proper combinations of thresholds
        """
        combinations = []  # List to store the combinations

        def helper(start, end, k, prefix):
            if k == 0:
                combinations.append(prefix)  # Add the combination to the list
                return
            for i in range(start, end - k + 2, step):
                helper(i + 1, end, k - 1, prefix + [i])

        helper(start, end, k, [])

        return combinations  # Return the list of combinations

    def multi_otsu(self, image, number_of_thresholds, step):
        """
        Performing image segmentation based on otsu algorithm

        Parameters:
            - image: The image to be thresholded
            - number_of_thresholds: the number of thresholds to seperate the histogram
            - step: the step taken by each threshold in each combination, not  by default for optimization purposes
        Returns:
            - otsu_img : The thresholded image
            - final_thresholds: A 2D array of the final thresholds containing only one element
            - separability_measure: A metric to evaluate the seperation process.
        """
        # Make a copy of the input image
        otsu_img = image.copy()
        # Create a pdf out of the input image
        pi_dist = Normalized_histogram_computation(otsu_img)
        # Initializing the maximum variance
        maximum_variance = 0
        # Get the list of the combination of all the candidate thresholds
        candidates_list = self.generate_combinations(
            start=1, end=255, k=number_of_thresholds, step=step
        )
        # Calculate the global mean to calculate the global variance to evaluate the seperation process
        global_mean = np.sum(np.arange(len(pi_dist)) * pi_dist)
        global_variance = np.sum(
            ((np.arange(len(pi_dist)) - global_mean) ** 2) * pi_dist
        )
        # Array to store the thresholds at which the between_class_variance has a maximum value
        threshold_values = []
        # Initialize to None
        separability_measure = None
        for candidates in candidates_list:
            # Compute the sum of probabilities for the first segment (from 0 to the first candidate)
            P_matrix = [np.sum(pi_dist[: candidates[0]])]
            # Compute the sum of probabilities for the middle segments
            P_matrix += [
                np.sum(pi_dist[candidates[i] : candidates[i + 1]])
                for i in range(len(candidates) - 1)
            ]
            # Compute the sum of probabilities for the last segment (from the last candidate to the end of the distribution)
            P_matrix.append(np.sum(pi_dist[candidates[-1] :]))
            # Check that no value in the sum matrix is zero
            if np.any(P_matrix) == 0:
                continue

            # Compute the mean value for the first segment
            if P_matrix[0] != 0:
                M_matrix = [
                    (1 / P_matrix[0])
                    * np.sum([i * pi_dist[i] for i in np.arange(0, candidates[0], 1)])
                ]
            else:
                M_matrix = [0]  # Handle division by zero
            # Compute the mean values for the middle segments
            M_matrix += [
                (
                    (1 / P_matrix[i + 1])
                    * np.sum(
                        [
                            ind * pi_dist[ind]
                            for ind in np.arange(candidates[i], candidates[i + 1], 1)
                        ]
                    )
                    if P_matrix[i + 1] != 0
                    else 0
                )
                for i in range(len(candidates) - 1)
            ]
            # Compute the mean value for the last segment
            M_matrix.append(
                (1 / P_matrix[-1])
                * np.sum(
                    [k * pi_dist[k] for k in np.arange(candidates[-1], len(pi_dist), 1)]
                )
                if P_matrix[-1] != 0
                else 0
            )
            # between_classes_variance = np.sum([P_matrix[0]*P_matrix[1]*((M_matrix[0] - M_matrix[1])**2) ])
            between_classes_variance = np.sum(
                [
                    P_matrix[i] * P_matrix[j] * ((M_matrix[i] - M_matrix[j]) ** 2)
                    for i, j in list(combinations(range(number_of_thresholds + 1), 2))
                ]
            )

            # Loop over all intensity levels and try them as thresholds, then compute the between_class_variance to check the separability measure according to this threshold value
            if between_classes_variance > maximum_variance:
                maximum_variance = between_classes_variance
                # If the between_class_variance corrisponding to this threshold intensity is maximum, store the threshold value
                threshold_values = [candidates]
                # Calculate the  Seperability Measure to evaluate the seperation process
                separability_measure = between_classes_variance / global_variance
            # To handel the case when there is more than one threshold value, maximize the between_class_variance, the optimal threshold in this case is their avg
            elif between_classes_variance == maximum_variance:
                threshold_values.append(candidates)
        # If there are multiple group of candidates, consider the mean value of each of them as the final thresholds
        if len(threshold_values) > 1:
            # Get the average of the thresholds that maximize the between_class_variance
            final_thresholds = [list(np.mean(threshold_values, axis=0, dtype=int))]
            # print('multi for the same threshold after averaging', final_thresholds)
        elif len(threshold_values) == 1:
            # if single threshold maximize the between_class_variance, then this is the perfect threshold to separate the classes
            # print('one for the max variance', threshold_values)
            final_thresholds = threshold_values
        else:
            # If no maximum between_class_variance then all the pixels belong to the same class (single intensity level), so assign them all to background class
            # we do so for simplicity since this often happen in the background pixels in the local thresholding, it rarely happen that the whole window has single
            # intensity in the object pixels
            otsu_img[np.where(image > 0)] = 255
            return otsu_img
        # Compute the regions in the image
        regions_in_image = [
            np.where(np.logical_and(image > 0, image < final_thresholds[0][0]))
        ]
        regions_in_image += [
            np.where(
                np.logical_and(
                    image > final_thresholds[0][i], image < final_thresholds[0][i + 1]
                )
            )
            for i in range(1, len(final_thresholds[0]) - 1)
        ]
        regions_in_image.append(np.where(image > final_thresholds[0][-1]))

        levels = np.linspace(0, 255, number_of_thresholds + 1)
        for i, region in enumerate(regions_in_image):
            otsu_img[region] = levels[i]
        return otsu_img, final_thresholds, separability_measure

    ## ============== Agglomerative Clustering ============== ##
    def get_agglomerative_parameters(self):
        self.downsampling = self.ui.downsampling.isChecked()
        self.agglo_number_of_clusters = self.ui.agglo_num_of_clusters_spinBox.value()
        self.agglo_scale_factor = self.ui.agglo_scale_factor.value()

    def downsample_image(self):
        # Get the dimensions of the original image
        height, width, channels = self.agglo_input_image.shape

        # Calculate new dimensions after downsampling
        new_width = int(width / self.agglo_scale_factor)
        new_height = int(height / self.agglo_scale_factor)

        # Create an empty array for the downsampled image
        downsampled_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        # Iterate through the original image and select pixels based on the scale factor
        for y in range(0, new_height):
            for x in range(0, new_width):
                downsampled_image[y, x] = self.agglo_input_image[
                    y * self.agglo_scale_factor, x * self.agglo_scale_factor
                ]

        return downsampled_image

    def agglo_reshape_image(self, image):
        pixels = image.reshape((-1, 3))
        return pixels

    def euclidean_distance(self, point1, point2):
        """
        Description:
            -   Computes euclidean distance of point1 and point2.
                Noting that "point1" and "point2" are lists.
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def clusters_distance(self, cluster1, cluster2):
        """
        Description:
            -   Computes distance between two clusters.
                cluster1 and cluster2 are lists of lists of points
        """
        return max(
            [
                self.euclidean_distance(point1, point2)
                for point1 in cluster1
                for point2 in cluster2
            ]
        )

    def clusters_distance_2(self, cluster1, cluster2):
        """
        Description:
            -   Computes distance between two centroids of the two clusters
                cluster1 and cluster2 are lists of lists of points
        """
        cluster1_center = np.average(cluster1, axis=0)
        cluster2_center = np.average(cluster2, axis=0)
        return self.euclidean_distance(cluster1_center, cluster2_center)

    def initial_clusters(self, points, initial_k=25):
        """
        partition pixels into self.initial_k groups based on color similarity
        """
        # Initialize a dictionary to hold the clusters each represented by:
        # the centroid color as a key
        # and the list of pixels that belong to that cluster as a value
        groups = {}
        # Defining the partitioning step
        d = int(256 / (initial_k))
        # Iterate over the range of initial clusters and assign the centroid colors for each cluster.
        # The centroid colors are determined by the multiples of the step size (d) ranging from 0 to 255.
        # Each centroid color is represented as an RGB tuple (j, j, j) where j is a multiple of d,
        # ensuring even distribution across the color space.
        for i in range(initial_k):
            j = i * d
            groups[(j, j, j)] = []
        # These lines iterate over each pixel in the image represented by the points array.
        # It calculates the Euclidean distance between the current pixel p and each centroid color (c)
        # using the euclidean_distance function. It then assigns the pixel p to the cluster with the closest centroid color.
        # The min function with a custom key function (lambda c: euclidean_distance(p, c)) finds the centroid color with the minimum distance to the pixel p,
        # and the pixel p is appended to the corresponding cluster in the groups dictionary.
        for i, p in enumerate(points):
            if i % 100000 == 0:
                print("processing pixel:", i)
            go = min(groups.keys(), key=lambda c: self.euclidean_distance(p, c))
            groups[go].append(p)
        # This line returns a list of pixel groups (clusters) where each group contains
        # the pixels belonging to that cluster.
        # It filters out any empty clusters by checking the length of each cluster list.
        return [g for g in groups.values() if len(g) > 0]

    def fit_clusters(self, points):

        # initially, assign each point to a distinct cluster
        print("Computing initial clusters ...")
        self.clusters_list = self.initial_clusters(points, initial_k=25)
        print("number of initial clusters:", len(self.clusters_list))
        print("merging clusters ...")

        while len(self.clusters_list) > self.agglo_number_of_clusters:

            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min(
                [
                    (c1, c2)
                    for i, c1 in enumerate(self.clusters_list)
                    for c2 in self.clusters_list[:i]
                ],
                key=lambda c: self.clusters_distance_2(c[0], c[1]),
            )

            # Remove the two clusters from the clusters list
            self.clusters_list = [
                c for c in self.clusters_list if c != cluster1 and c != cluster2
            ]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

            print("number of clusters:", len(self.clusters_list))

        print("assigning cluster num to each point ...")
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        print("Computing cluster centers ...")
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def predict_cluster(self, point):
        """
        Find cluster number of point
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        """
        Find center of the cluster that point belongs to
        """
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center

    def apply_agglomerative_clustering(self):
        start = time.time()
        if self.downsampling:
            agglo_downsampled_image = self.downsample_image()
        else:
            agglo_downsampled_image = self.agglo_input_image
        self.get_agglomerative_parameters()
        pixels = self.agglo_reshape_image(agglo_downsampled_image)
        self.fit_clusters(pixels)

        self.agglo_output_image = [
            [self.predict_center(pixel) for pixel in row]
            for row in agglo_downsampled_image
        ]
        self.agglo_output_image = np.array(self.agglo_output_image, np.uint8)

        self.display_image(
            self.agglo_output_image,
            self.ui.agglomerative_output_figure_canvas,
            f"Segmented image with k={self.agglo_number_of_clusters}",
            False,
        )

        end = time.time()
        elapsed_time_seconds = end - start
        minutes = int(elapsed_time_seconds // 60)
        seconds = int(elapsed_time_seconds % 60)
        self.ui.agglo_elapsed_time.setText(
            "Elapsed Time is {:02d} minutes and {:02d} seconds".format(minutes, seconds)
        )


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = BackendClass()
    MainWindow.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())

# backend.py
import os

import numpy as np

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import time

import cv2
import numpy as np

# in CMD: pip install qdarkstyle -> pip install pyqtdarktheme
import qdarktheme
from FeatureCraft_UI import FeatureCraft_Ui
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from harris_utils import *
from helper_functions import *
from keypoint_descriptor import *


class BackendClass(QMainWindow, FeatureCraft_Ui):
    def __init__(self):
        super().__init__()
        self.ui = FeatureCraft_Ui()
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
        self.ui.actionImport_Image.triggered.connect(self.load_image)

        # Set the icon and title
        self.change_the_icon()

    def change_the_icon(self):
        self.setWindowIcon(QtGui.QIcon("assets/icons/app_icon.png"))
        self.setWindowTitle("exVision-FeatureCraft")

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
            self.ui.histogram_global_thresholds_label.setText(" ")
            if grey:
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )
            else:
                image = convert_to_gray(image)
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )

        ax.axis(axis_disabled)
        ax.set_title(title)
        canvas.figure.subplots_adjust(left=0.1, right=0.90, bottom=0.08, top=0.95)
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
                    self.ui.sift_target_figure_canvas,
                    "Target Image",
                    False,
                )
            elif response == QMessageBox.No:
                self.sift_template_image = image
                self.display_image(
                    image,
                    self.ui.sift_template_figure_canvas,
                    "Template Image",
                    False,
                )

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
            gray = convert_to_gray(img_RGB)
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
        gray = convert_to_gray(img_RGB)
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

    def apply_sift(self):
        start = time.time()
        self.sift_target_image, ratio = sift_resize(self.sift_target_image)
        self.sift_template_image, _ = sift_resize(self.sift_template_image, ratio)

        img_kp, img_des = computeKeypointsAndDescriptors(
            self.sift_target_image,
            self.n_octaves,
            self.s_value,
            self.sigma_base,
            self.contrast_th,
            self.r_ratio,
        )
        template_kp, template_des = computeKeypointsAndDescriptors(
            self.sift_template_image,
            self.n_octaves,
            self.s_value,
            self.sigma_base,
            self.contrast_th,
            self.r_ratio,
        )

        img_match = match(
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


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = BackendClass()
    MainWindow.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())

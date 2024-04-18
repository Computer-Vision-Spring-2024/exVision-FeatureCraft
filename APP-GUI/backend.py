# backend.py
import os
# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# imports
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QApplication, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtGui
from task3_ui import Ui_MainWindow  
# from scipy.signal import convolve2d
import cv2
import numpy as np
import time
from matplotlib.figure import Figure


# Helper functions
def convert_to_grey( img_RGB: np.ndarray) -> np.ndarray:
        grey = np.dot(img_RGB[..., :3], [0.2989, 0.5870, 0.1140])
        return grey

def convert_BGR_to_RGB(img_BGR_nd_arr : np.ndarray) -> np.ndarray: 
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
        numpy.ndarray: The padded matrix.
    """
    padded_matrix = np.zeros((height + 2 * pad_size, width + 2 * pad_size)) # zeros matrix 
    padded_matrix[pad_size:pad_size+height, pad_size:pad_size+width] = matrix  
    return padded_matrix



def convolve2d_optimized(input_matrix, convolution_kernel, mode='same'):
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
    padded_matrix = padding_matrix(input_matrix, input_width, input_height, pad_size=padding_size)

    # Create an array of offsets for convolution
    offset_array = np.arange(-padding_size, padding_size + 1)

    # Create a meshgrid of indices for convolution
    x_indices, y_indices = np.meshgrid(offset_array, offset_array, indexing='ij')

    # Add the meshgrid indices to an array of the original indices
    i_indices = np.arange(padding_size, input_height + padding_size)[:, None, None] + x_indices.flatten()
    j_indices = np.arange(padding_size, input_width + padding_size)[None, :, None] + y_indices.flatten()

    # Use advanced indexing to get the regions for convolution
    convolution_regions = padded_matrix[i_indices, j_indices].reshape(input_height, input_width, kernel_size, kernel_size)

    # Compute the convolution by multiplying the regions with the kernel and summing the results
    output_matrix = np.sum(convolution_regions * convolution_kernel, axis=(2, 3))

    return output_matrix


class BackendClass(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initials
        self.current_image_RGB = None
        self.r = None
        self.eigenvalues = None
        self.change_the_icon()

        # The threshold slider in corner detection tab
        self.ui.horizontalSlider_corner_tab.setEnabled(False)
        

        # Connect menu action to load_image
        self.ui.actionLoad_Image.triggered.connect(self.load_image)
        self.ui.apply_harris_push_button.clicked.connect(lambda: self.on_apply_detectors_clicked(self.current_image_RGB, 0))
        self.ui.apply_lambda_minus_push_button.clicked.connect(lambda : self.on_apply_detectors_clicked(self.current_image_RGB,1))

        # Create QVBoxLayouts for left_widget_corner_tab and right_widget_corner_tab
        self.left_layout = QVBoxLayout(self.ui.left_widget_corner_tab)
        self.right_layout = QVBoxLayout(self.ui.right_widget_corner_tab)

    
    def change_the_icon(self):
        self.setWindowIcon(QtGui.QIcon('App_Icon.png'))
        self.setWindowTitle("Computer Vision - Task 03 - Team 02")

    def load_image(self, file_path=None, folder=""):
        # clear self.r and threshold label
        self.ui.threshold_value_label.setText("")
        self.r = None
        self.eigenvalues = None
        
        # print(file_path) # For some reason it's False
        # Get the path of the image
        if file_path is False:
            # Open file dialog if file_path is not provided
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Image",
                folder,
                "Image Files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm)",
            )

        # Check if file_path is not None and it's a string
        if file_path and isinstance(file_path, str):
            # Read the matrix, convert to rgb
            img = cv2.imread(file_path, 1)
            
            self.current_image_RGB = convert_BGR_to_RGB(img)
            # Create instance of 
            fig_left = Figure()
            # Add subplot to the figure, ax
            ax_left = fig_left.add_subplot(111)
            # ax.imshow
            ax_left.imshow(self.current_image_RGB)
            ax_left.axis('off')  # Turn off axis
            # Create figure canvas from figure
            canvas_left = FigureCanvas(fig_left)
            # Clear existing layouts before adding canvas
            for i in reversed(range(self.left_layout.count())):
                self.left_layout.itemAt(i).widget().setParent(None)
            # Add the canvases to the layouts
            self.left_layout.addWidget(canvas_left)
            self.clear_right_image()
            # Deactivate the slider and disconnect from apply harris function
            self.ui.horizontalSlider_corner_tab.setEnabled(False)
            try:
                self.ui.horizontalSlider_corner_tab.valueChanged.disconnect()
            except TypeError:
                pass
            
    def on_apply_detectors_clicked(self, img_RGB,operator):
        if self.current_image_RGB.any():
            self.ui.horizontalSlider_corner_tab.valueChanged.connect(lambda value: self.on_changing_threshold(value, img_RGB,operator))

            if operator == 0:
                # call the function with default parameters
                start = time.time()
                self.apply_harris_detector_vectorized(img_RGB)
                finish = time.time()
                self.ui.consumed_time_label.setText(f"This Operation consumed {finish-start:.3f} seconds || ")
                # Activate the slider and connect with applya harris function
                self.ui.horizontalSlider_corner_tab.setEnabled(True)
                self.ui.horizontalSlider_corner_tab.setMinimum(1)
                self.ui.horizontalSlider_corner_tab.setMaximum(int(10E6))
                self.ui.horizontalSlider_corner_tab.setSingleStep(10000)
                self.ui.horizontalSlider_corner_tab.setValue(10000)
                self.ui.threshold_value_label.setText(str(10000))
            elif operator == 1:
                # call the function with default parameters
                start = time.time()
                self.apply_lambda_minus_vectorized(img_RGB)
                finish = time.time()
                self.ui.consumed_time_label.setText(f"This Operation consumed {finish-start:.3f} seconds || ")
                # Activate the slider and connect with apply lambda function
                self.ui.horizontalSlider_corner_tab.setEnabled(True)
                self.ui.horizontalSlider_corner_tab.setMinimum(1)  
                self.ui.horizontalSlider_corner_tab.setMaximum(10000)  
                self.ui.horizontalSlider_corner_tab.setSingleStep(1)  
                self.ui.horizontalSlider_corner_tab.setValue(10)  

                self.ui.threshold_value_label.setText(f"{0.01}% of max eigen value")
        return
    
    def on_changing_threshold(self,threshold, img_RGB, operator):
        output_img = img_RGB.copy()
        if operator == 0:
            if np.all(self.r != None):
                # Show the slider value using a label
                self.ui.threshold_value_label.setText(str(threshold))
                # Apply threshold and store detected corners
                corner_list = np.argwhere(self.r > threshold)
                # Create output image
                
                output_img[corner_list[:, 0], corner_list[:, 1]] = (255, 0, 0)  # Highlight detected corners in red
                self.display_right_image(output_img)
            elif operator == 1:
                if np.all(self.eigenvalues != None):
                    # Set the value of the threshold
                    value = self.ui.horizontalSlider_corner_tab.value() / 10000.0

                    # Show the slider value using a label
                    self.ui.threshold_value_label.setText(f"{value}% of max eigen value")
                    # Apply threshold and store detected corners
                    corners = np.where(self.eigenvalues > value)

                    # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
                    for i, j in zip(*corners):
                        cv2.circle(output_img, (j, i), 3, (0, 255, 0), -1)  # Green color
                    self.display_right_image(output_img)

        return

    def apply_harris_detector_vectorized(self,img_RGB, window_size=5, k=0.04, threshold=10000):
        if np.all(img_RGB != None):
            # Convert image to grayscale
            gray = convert_to_grey(img_RGB)

            # Compute image gradients using prewitt 5x5 operator
            # K_X = np.array([[-1, -2, 0, 2, 1],
            #     [-2, -3, 0, 3, 2],
            #     [-3, -5, 0, 5, 3],
            #     [-2, -3, 0, 3, 2],
            #     [-1, -2, 0, 2, 1]])

            # K_Y = K_X.T  # The kernel for vertical edges is the transpose of the kernel for horizontal edges

            # scale_factor = 1
            # Ix, Iy = scale_factor * convolve2d_optimized(gray, K_X, mode='same'),scale_factor * convolve2d_optimized(gray, K_Y, mode='same') 
            Ix, Iy = np.gradient(gray)
            # Compute products of derivatives
            Ixx = Ix**2
            Ixy = Iy*Ix
            Iyy = Iy**2

            # Define window function
            window = np.ones((window_size, window_size))

            # Compute sums of the second moment matrix over the window
            Sxx = convolve2d_optimized(Ixx, window, mode='same')
            Sxy = convolve2d_optimized(Ixy, window, mode='same')
            Syy = convolve2d_optimized(Iyy, window, mode='same')

            # Compute determinant and trace of the second moment matrix
            det = Sxx*Syy - Sxy**2
            trace = Sxx + Syy

            # Compute corner response
            self.r = det - k*(trace**2)

            # Apply threshold and store detected corners
            corner_list = np.argwhere(self.r > threshold)
            corner_response = self.r[self.r > threshold]

            # Create output image
            output_img = img_RGB.copy()
            output_img[corner_list[:, 0], corner_list[:, 1]] = (0, 0, 255)  # Highlight detected corners in blue
            self.display_right_image(output_img)

            return list(zip(corner_list[:, 1], corner_list[:, 0], corner_response)), output_img

    def apply_lambda_minus_vectorized(self, img_RGB, window_size = 5, threshold_percentage = 0.01):
            
        # Convert image to grayscale
            gray = convert_to_grey(img_RGB)
            output_image = img_RGB.copy()
            # Compute the gradient using Sobel 5x5 operator
            K_X = np.array([[-1, -2, 0, 2, 1],
                            [-2, -3, 0, 3, 2],
                            [-3, -5, 0, 5, 3],
                            [-2, -3, 0, 3, 2],
                            [-1, -2, 0, 2, 1]])

            K_Y = K_X.T  # The kernel for vertical edges is the transpose of the kernel for horizontal edges

            gradient_x, gradient_y = convolve2d_optimized(gray, K_X, mode='same'),convolve2d_optimized(gray, K_Y, mode='same') 
            # Compute the elements of the H matrix
            H_xx = gradient_x * gradient_x
            H_yy = gradient_y * gradient_y
            H_xy = gradient_x * gradient_y
            # Compute the sum of the elements in a neighborhood (e.g., using a Gaussian kernel)
            # Define window function
            window = np.ones((5, 5))
            H_xx_sum = convolve2d_optimized(H_xx, window, mode='same')/25
            H_yy_sum = convolve2d_optimized(H_yy, window, mode='same')/25
            H_xy_sum = convolve2d_optimized(H_xy, window, mode='same')/25

            # Compute the eigenvalues
            H = np.stack([H_xx_sum, H_xy_sum, H_xy_sum, H_yy_sum], axis=-1).reshape(-1, 2, 2)
            self.eigenvalues = np.linalg.eigvalsh(H).min(axis=-1).reshape(gray.shape)

            # Threshold to find corners
            threshold = threshold_percentage * self.eigenvalues.max() 
            corners = np.where(self.eigenvalues > threshold)
            
            # Draw circles at detected corners by unpacking the corner object, drawing at each corner and then restoring its original combact state
            for i, j in zip(*corners):
                cv2.circle(output_image, (j, i), 3, (0, 255, 0), -1)  # Green color
            self.display_right_image(output_image)


    def display_right_image(self, img_RGB):
        """
        img is an RGB matrix
        """
        if img_RGB.any() == None:
            pass
        else:
            if self.right_layout.count() > 0:
                # Get the existing canvas widget
                canvas_right = self.right_layout.itemAt(0).widget()
                # Update the figure displayed on the canvas
                ax_right = canvas_right.figure.axes[0]
                ax_right.clear()
                ax_right.imshow(img_RGB)
                ax_right.axis('off')  # Turn off axis
                # Redraw the canvas
                canvas_right.draw()
            else:
                fig_right = Figure()
                ax_right = fig_right.add_subplot(111)
                ax_right.axis('off')  # Turn off axis
                ax_right.imshow(img_RGB)
                canvas_right = FigureCanvas(fig_right)
                self.right_layout.addWidget(canvas_right)

    
    def clear_right_image(self):
        # Clear existing layouts before adding canvas
        for i in reversed(range(self.right_layout.count())):
            widget = self.right_layout.itemAt(i).widget()
            # Remove it from the layout list
            self.right_layout.removeWidget(widget)
            # Remove the widget from the GUI
            widget.setParent(None)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = BackendClass()
    MainWindow.show()
    sys.exit(app.exec_())



# backend.py
import os
# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# imports
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow, QApplication, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import numpy as np
from task3_ui import Ui_MainWindow  
# from scipy.signal import convolve2d


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

        # The threshold slider in corner detection tab
        self.ui.horizontalSlider_corner_tab.setEnabled(False)
        self.ui.horizontalSlider_corner_tab.setMinimum(100)
        self.ui.horizontalSlider_corner_tab.setMaximum(int(10E6))
        self.ui.horizontalSlider_corner_tab.setSingleStep(10000)
        self.ui.horizontalSlider_corner_tab.setValue(10000)

        # Connect menu action to load_image
        self.ui.actionLoad_Image.triggered.connect(self.load_image)
        self.ui.apply_harris_push_button.clicked.connect(lambda: self.on_apply_harris_button_clicked(self.current_image_RGB))

        # Create QVBoxLayouts for left_widget_corner_tab and right_widget_corner_tab
        self.left_layout = QVBoxLayout(self.ui.left_widget_corner_tab)
        self.right_layout = QVBoxLayout(self.ui.right_widget_corner_tab)

    

    def load_image(self, file_path=None, folder=""):
        # clear self.r and threshold label
        self.ui.threshold_value_label.setText("")
        self.r = None
        
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
            
    def on_apply_harris_button_clicked(self, img_RGB):

        # call the function with default parameters
        self.apply_harris_detector_vectorized(img_RGB)
        # Activate the slider and connect with applya harris function
        self.ui.horizontalSlider_corner_tab.setEnabled(True)
        self.ui.horizontalSlider_corner_tab.setValue(10000)
        self.ui.threshold_value_label.setText(str(10000))
        self.ui.horizontalSlider_corner_tab.valueChanged.connect(lambda value: self.on_changing_threshold(value, img_RGB=img_RGB))
        return

    def apply_harris_detector(self, img_RGB, window_size = 5, k = 0.04, threshold = 10000):
        """
        Detect corners in an image using the Harris corner detection algorithm.

        Args:
        - img_path (str): Path to the input image file.
        - window_size (int): Size of the window for computing the corner response function.
        - k (float): Harris detector free parameter in the range [0.04, 0.06].
        - threshold (float): Threshold value for corner detection.

        Returns:
        - corner_list (list): List of detected corners, each represented as [x, y, r] where x and y are the coordinates
                            of the corner and r is the corner response value.
        - output_img (numpy.ndarray): Image with detected corners highlighted in red.
        """
        corner_list = []  # List to store detected corners
        img = img_RGB
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = convert_to_grey(img_RGB)   # Convert image to grayscale
        output_img = img # cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB)  # Convert grayscale image to RGB for visualization 

        height = img.shape[0]  # Get height of the image
        width = img.shape[1]   # Get width of the image

        # Compute image gradients
        dy, dx = np.gradient(gray)

        # Compute products and second derivatives
        Ixx = dx**2
        Ixy = dy*dx
        Iyy = dy**2

        offset = int(window_size / 2)
        y_range = height - offset
        x_range = width - offset

        print("Finding Corners...")
        for y in range(offset, y_range):
            for x in range(offset, x_range):
                start_y = y - offset
                end_y = y + offset + 1
                start_x = x - offset
                end_x = x + offset + 1

                # Extract windows for second moment matrix
                windowIxx = Ixx[start_y:end_y, start_x:end_x]
                windowIxy = Ixy[start_y:end_y, start_x:end_x]
                windowIyy = Iyy[start_y:end_y, start_x:end_x]

                # Compute sums of the second moment matrix
                Sxx = windowIxx.sum()
                Sxy = windowIxy.sum()
                Syy = windowIyy.sum()

                # Compute determinant and trace of the second moment matrix
                det = (Sxx * Syy) - (Sxy**2)
                trace = Sxx + Syy

                # Compute corner response
                self.r = det - k * (trace**2)

                # Apply threshold and store detected corners
                if self.r > threshold:
                    corner_list.append([x, y, r])
                    output_img[y, x] = (255, 0, 0)  # Highlight detected corners in red
        self.display_right_image(output_img)
        return 
    
    def on_changing_threshold(self,threshold, img_RGB):
        
        if np.all(self.r != None):
            # Show the slider value using a label
            self.ui.threshold_value_label.setText(str(threshold))
            # Apply threshold and store detected corners
            corner_list = np.argwhere(self.r > threshold)
            # Create output image
            output_img = img_RGB.copy()
            output_img[corner_list[:, 0], corner_list[:, 1]] = (255, 0, 0)  # Highlight detected corners in red
            self.display_right_image(output_img)
        return

    def apply_harris_detector_vectorized(self,img_RGB, window_size=5, k=0.04, threshold=10000):
        if np.all(img_RGB != None):
            # Convert image to grayscale
            gray = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)

            # Compute image gradients
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
            output_img[corner_list[:, 0], corner_list[:, 1]] = (255, 0, 0)  # Highlight detected corners in red
            self.display_right_image(output_img)

            return list(zip(corner_list[:, 1], corner_list[:, 0], corner_response)), output_img

    def display_right_image(self, img):
        """
        img is an RGB matrix
        """
        if img.any() == None:
            pass
        else:
            if self.right_layout.count() > 0:
                # Get the existing canvas widget
                canvas_right = self.right_layout.itemAt(0).widget()
                # Update the figure displayed on the canvas
                ax_right = canvas_right.figure.axes[0]
                ax_right.clear()
                ax_right.imshow(img)
                ax_right.axis('off')  # Turn off axis
                # Redraw the canvas
                canvas_right.draw()
            else:
                fig_right = Figure()
                ax_right = fig_right.add_subplot(111)
                ax_right.axis('off')  # Turn off axis
                ax_right.imshow(img)
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



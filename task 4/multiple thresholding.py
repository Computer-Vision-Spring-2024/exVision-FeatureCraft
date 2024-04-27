import numpy as np
import cv2
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

def apply_local_threshold(grayscale_image, threshold_algorithm, kernel_size = 5):
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
    #Pad the image to avoid lossing information of the boundry pixels or getting out of bounds 
    padded_image = _pad_image(kernel_size, grayscale_image)
    thresholded_image = np.zeros_like(grayscale_image)
    for i in range(grayscale_image.shape[0]):
        for j in range(grayscale_image.shape[1]):
            # Take the current pixel and its neighboors to apply the thresholding algorithm on them
            window = padded_image[i : i + kernel_size, j : j + kernel_size]
            # If all the pixels belong to the same class (single intensity level), assign them all to background class 
            # we do so for simplicity since this often happen in the background pixels in the local thresholding, it rarely happen that the whole window has single
            # intensity in the object pixels 
            if np.all(window == window[0, 0]):
                thresholded_image[i, j] =255
            else:
                # Assign the value of the middle pixel of the thresholded window to the current pixel of the thresholded image
                thresholded_image[i: i+kernel_size//2, j: j+kernel_size//2] = threshold_algorithm(window)[:kernel_size//2 ,:kernel_size//2]
    return thresholded_image

def optimal_thresholding(image):
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
    background_mean=np.sum(corners)/4
    # Calculate the mean of the object class by summing the intensities of the image then subtracting the four corners then dividing by the number 
    # of pixels in the full image - 4 
    object_mean= (np.sum(image) - np.sum(corners)) / (image.shape[0] * image.shape[1] - 4)
    # Set random iinitial values for the thresholds 
    threshold=-1
    prev_threshold=0
    # keep updating the threshold based on the means of the two classes until the new threshold equals the previous one
    while (abs(threshold- prev_threshold)) > 0:
        # Store the threshold value before updating it to compare it to the new one in the next iteration
        prev_threshold= threshold
        # Compute the new threshold value midway between the two means of the two classes 
        threshold= (background_mean+ object_mean)/2
        # Get the indices whose intensity values are less than the threshold 
        background_pixels= np.where(image < threshold)
        # Get the indices whose intensity values are more than the threshold 
        object_pixels= np.where(image> threshold)
        if not len(background_pixels[0])==0:
            # Compute the new mean of the background class based on the new threshold 
            background_mean= np.sum(image[background_pixels])/ len(background_pixels[0])
        if not len(object_pixels[0])==0:
            # Compute the new mean of the object class based on the new threshold 
            object_mean= np.sum(image[object_pixels])/len(object_pixels[0])
    # Set background pixels white 
    image[background_pixels]=0
    # Set object pixels black
    image[object_pixels]= 255
    return(image)

def otsu_thresholding(image):
    """
    Description:
        - Applies Otsu thresholding to an image.

    Args:
        - image: the image to be thresholded

    Returns:
        - [numpy ndarray]: the resulted thresholded image after applying Otsu threshoding algorithm.
    """
    # Calculate the normalized histogram of the image 
    normalized_histogram= Normalized_histogram_computation(image)
    # initialize the weighted sum to be equal 0, which corrisponds to the weighted sum of the zero intensity (0* P(0))
    weighted_sum= 0
    # initialize the probability of class one to be equal to the probability of the 0 intensity
    probability_class1=normalized_histogram[0]
    # Calculate the mean of the image 
    global_mean = np.sum(np.arange(len(normalized_histogram)) * normalized_histogram)
    # Calculate the variance of the image
    global_variance = np.sum(((np.arange(len(normalized_histogram)) - global_mean)**2) * normalized_histogram)
    # Variable to track the maximum between_class_variance achieved through different thresholds 
    maximum_variance= 0
    # Array to store the thresholds at which the between_class_variance has a maximum value
    threshold_values=[]
    # Loop over all intensity levels and try them as thresholds, then compute the between_class_variance to check the separability measure according to this threshold value
    for k in range(1, 256):
        # The probability of class1 is calculated through the cumulative sum of the probabilities of all the intensities smaller than or equal the threshold
        probability_class1 += normalized_histogram[k]
        weighted_sum+= k*normalized_histogram[k]
        # if probability of class 1 equals zero or one, this means that according to the current threshold there is a single class, then there is no between_class_variance
        if probability_class1*(1-probability_class1) ==0: continue
        # This form for calculating the between_class_variance is obtained from substituting with those two equations: P1+P2=1, P1 *m1 +P2 * m2= mg
        # in the ordinary form for calculating the between_class_variance ( between_class_variance= P1*P2 (m1-m2)**2)
        # this form is slightly more efficient computationally than the ordinary form 
        # because the global mean, mG, is computed only once, so only two parameters, weighted_sum and probability_class1, need to be computed for any value of k.
        between_class_variance= (((global_mean* probability_class1) -weighted_sum)**2)/(probability_class1*(1-probability_class1))
        if between_class_variance > maximum_variance:
            maximum_variance= between_class_variance
            # If the between_class_variance corrisponding to this threshold intensity is maximum, store the threshold value 
            threshold_values=[k]
            # after connecting the backend with the UI this is recommended to be attribute to self, in order to display its value in a line edit in the UI
            separability_measure= between_class_variance/ global_variance
            # To handel the case when there is more than one threshold value, maximize the between_class_variance, the optimal threshold in this case is their avg
        elif between_class_variance == maximum_variance:
            threshold_values.append(k)
    if len(threshold_values)>1:
        # Get the average of the thresholds that maximize the between_class_variance
        threshold= np.mean(threshold_values)
    elif len(threshold_values)==1:
        # if single threshold maximize the between_class_variance, then this is the perfect threshold to separate the classes
        threshold= threshold[0]
    else:
        # If no maximum between_class_variance then all the pixels belong to the same class (single intensity level), so assign them all to background class 
        # we do so for simplicity since this often happen in the background pixels in the local thresholding, it rarely happen that the whole window has single
        # intensity in the object pixels 
        image[np.where(image>0)]= 255
        return image
    background_pixels= np.where(image < threshold)
    object_pixels= np.where(image> threshold)
    image[background_pixels]=0
    image[object_pixels]= 255      
    return image     
        

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
    Histogram /= (Image_Height * Image_Width)
                
    return Histogram


image= cv2.imread(r'E:\year 4\computer vision\task2\ImageAlchemy\App\Resources\HoughSamples\Circle\Coins01.jpeg',0)
thresholded= apply_local_threshold(image, optimal_thresholding, 5)
# thresholded= otsu_thresholding(image)
# thresholded= apply_local_threshold(image, otsu_thresholding,5)
# thresholded= optimal_thresholding(image)
cv2.imshow(' image',thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
    
    
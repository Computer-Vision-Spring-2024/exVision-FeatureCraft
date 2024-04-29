from itertools import combinations

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
    for i in range(0,grayscale_image.shape[0],4):
        for j in range(0,grayscale_image.shape[1],4):
            # Take the current pixel and its neighboors to apply the thresholding algorithm on them
            window = padded_image[i : i + kernel_size, j : j + kernel_size]
            # If all the pixels belong to the same class (single intensity level), assign them all to background class 
            # we do so for simplicity since this often happen in the background pixels in the local thresholding, it rarely happen that the whole window has single
            # intensity in the object pixels 
            if np.all(window == window[0, 0]):
                thresholded_image[i, j] =255
                thresholded_window = window
            else:
                # Assign the value of the middle pixel of the thresholded window to the current pixel of the thresholded image
                thresholded_window, _ = threshold_algorithm(window)
                thresholded_image[i: i+kernel_size//2, j: j+kernel_size//2] = thresholded_window[:kernel_size//2 ,:kernel_size//2]
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
    return(image, threshold)


     

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
    # print('hist in image',len( Histogram))
                
    return Histogram

def generate_combinations( k, step ,start = 1, end = 255):
    ''''' 
        Generate proper combinations of thresholds for histogram bins based on the number of thresholds

        Parameters:
        - k: the number of thresholds
        - step: the increment distance for the threshold, not 1 by default for optimization purposes
        - start: the first number in histogram bins which is 1 by default.
        - end: last number in histogram bins which is 255 by default.

        Returns:
        a list of the proper combinations of thresholds
    '''''
    combinations = []  # List to store the combinations

    def helper(start, end, k, prefix):
        if k == 0:
            combinations.append(prefix)  # Add the combination to the list
            return
        for i in range(start, end-k+2, step):
            helper(i+1, end, k-1, prefix + [i])

    helper(start, end, k, [])

    return combinations  # Return the list of combinations

# Test the function

def multi_otsu(image, number_of_thresholds, step):
    ''''
    Performing image segmentation based on otsu algorithm

    Parameters:
        - image: The image to be thresholded
        - number_of_thresholds: the number of thresholds to seperate the histogram
        - step: the step taken by each threshold in each combination, not  by default for optimization purposes
    Returns:
        - otsu_img : The thresholded image
        - final_thresholds: A 2D array of the final thresholds containing only one element
        - separability_measure: A metric to evaluate the seperation process.
    '''
    # Make a copy of the input image
    otsu_img = image.copy()
    # Create a pdf out of the input image
    pi_dist = Normalized_histogram_computation(otsu_img)
    # Initializing the maximum variance 
    maximum_variance= 0
    # Get the list of the combination of all the candidate thresholds
    candidates_list = generate_combinations(start=1, end=255, k=number_of_thresholds, step=step)
    # Calculate the global mean to calculate the global variance to evaluate the seperation process
    global_mean = np.sum(np.arange(len(pi_dist)) * pi_dist)
    global_variance = np.sum(((np.arange(len(pi_dist)) - global_mean)**2) * pi_dist)
    # Array to store the thresholds at which the between_class_variance has a maximum value
    threshold_values=[]
    # Initialize to None
    separability_measure = None  
    for candidates in candidates_list:
        # Compute the sum of probabilities for the first segment (from 0 to the first candidate)
        P_matrix = [np.sum(pi_dist[:candidates[0]])]
        # Compute the sum of probabilities for the middle segments
        P_matrix += [np.sum(pi_dist[candidates[i]:candidates[i+1]]) for i in range(len(candidates)-1)]
        # Compute the sum of probabilities for the last segment (from the last candidate to the end of the distribution)
        P_matrix.append(np.sum(pi_dist[candidates[-1]:]))
        # Check that no value in the sum matrix is zero
        if np.any(P_matrix) == 0:
            continue
        
        # Compute the mean value for the first segment
        if P_matrix[0] != 0:
            M_matrix = [(1/P_matrix[0]) * np.sum([i*pi_dist[i] for i in np.arange(0,candidates[0],1)])]
        else:
            M_matrix = [0]  # Handle division by zero
        # Compute the mean values for the middle segments
        M_matrix += [(1/P_matrix[i+1]) * np.sum([ind*pi_dist[ind] for ind in np.arange(candidates[i],candidates[i+1],1)]) if P_matrix[i+1] != 0 else 0 for i in range(len(candidates)-1)]
        # Compute the mean value for the last segment
        M_matrix.append((1/P_matrix[-1]) * np.sum([k*pi_dist[k] for k in np.arange(candidates[-1], len(pi_dist), 1)]) if P_matrix[-1] != 0 else 0)
        # between_classes_variance = np.sum([P_matrix[0]*P_matrix[1]*((M_matrix[0] - M_matrix[1])**2) ])
        between_classes_variance = np.sum([P_matrix[i]*P_matrix[j]*((M_matrix[i] - M_matrix[j])**2) for i, j in list(combinations(range(number_of_thresholds+1),2))])

        # Loop over all intensity levels and try them as thresholds, then compute the between_class_variance to check the separability measure according to this threshold value
        if between_classes_variance > maximum_variance:
            maximum_variance= between_classes_variance
            # If the between_class_variance corrisponding to this threshold intensity is maximum, store the threshold value 
            threshold_values= [candidates]
            # Calculate the  Seperability Measure to evaluate the seperation process
            separability_measure = between_classes_variance / global_variance
        # To handel the case when there is more than one threshold value, maximize the between_class_variance, the optimal threshold in this case is their avg
        elif between_classes_variance == maximum_variance:
            threshold_values.append(candidates)
    # If there are multiple group of candidates, consider the mean value of each of them as the final thresholds
    if len(threshold_values)>1:
        # Get the average of the thresholds that maximize the between_class_variance
        final_thresholds =[list( np.mean(threshold_values, axis=0, dtype=int))]
        # print('multi for the same threshold after averaging', final_thresholds)
    elif len(threshold_values)==1:
        # if single threshold maximize the between_class_variance, then this is the perfect threshold to separate the classes
        # print('one for the max variance', threshold_values)
        final_thresholds = threshold_values
    else:
        # If no maximum between_class_variance then all the pixels belong to the same class (single intensity level), so assign them all to background class 
        # we do so for simplicity since this often happen in the background pixels in the local thresholding, it rarely happen that the whole window has single
        # intensity in the object pixels 
        otsu_img[np.where(image>0)]= 255
        return otsu_img
    # Compute the regions in the image
    regions_in_image = [np.where(np.logical_and(image > 0, image < final_thresholds[0][0]))]
    regions_in_image += [np.where(np.logical_and(image > final_thresholds[0][i], image < final_thresholds[0][i+1])) for i in range(1, len(final_thresholds[0])-1)]
    regions_in_image.append(np.where(image > final_thresholds[0][-1]))

    levels = np.linspace(0,255,number_of_thresholds+1)
    for i,region in enumerate(regions_in_image):
        otsu_img[region] = levels[i]
    return otsu_img, final_thresholds, separability_measure
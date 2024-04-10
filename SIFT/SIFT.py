import numpy as np
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def gaussian_filter_kernel( sigma, kernel_size= 11):
        """
        Description:
            - Generates a Gaussian filter kernel.

        Args:
            - kernel_size: Size of the square kernel (e.g., 3x3).
            - sigma: Standard deviation of the Gaussian distribution.

        Returns:
            - A numpy array representing the Gaussian filter kernel.
        """
        offset = kernel_size // 2

        x = np.arange(-offset, offset + 1)[:, np.newaxis]
        y = np.arange(-offset, offset + 1)
        x_squared = x**2
        y_squared = y**2

        kernel = np.exp(-(x_squared + y_squared) / (2 * sigma**2))
        kernel /= 2 * np.pi * (sigma**2)  # for normalization

        return kernel
    
def generateGaussianKernels(sigma, num_intervals):
    """
        Description:
            - Generates the required Gaussian Kernels for generating different scales for each octave.

        Args:
            - sigma: Standard deviation of the Gaussian distribution.
            - num_intervals: the order of the image that is blurred with 2 sigma

        Returns:
            - gaussian_kernels: A numpy array of arrays in which the generated Gaussian kernels are stored.
    """
    gaussian_kernels= []
    scale_level= sigma 
    # To cover a complete octave, we need 's + 3' blurred images. This ensures that we have enough information for accurate feature detection.
    images_per_octave = num_intervals + 3 
    # constant multiplicative factor k which separates two nearby scales in the octave
    k = 2 ** (1 / num_intervals)
    gaussian_kernels.append(gaussian_filter_kernel(sigma))
    # generate kernel for each image in the octave
    for iterator in range(1, images_per_octave):
        # multiply the current scale level with the multiplicative factor
        scale_level *= k 
        gaussian_kernels.append(gaussian_filter_kernel(scale_level))
    return gaussian_kernels

def pad_image(image, kernel_size=11):
        """
        Description:
            - Pads the image with zeros.

        Returns:
            - [numpy.ndarray]: A padded image.
        """
        pad_width = kernel_size // 2
        return np.pad(
            image,
            ((pad_width, pad_width), (pad_width, pad_width)),
            mode="edge",
        )
def generate_octaves_pyramid(img, t_c, R_TH, num_octaves=4, intervals= 2):
    """
        Description:
            - Generates the gaussian pyramid which consists of several octaves with increasingly blurred images.

        Args:
            - img: the image whose features should be extracted.
            - t_c: the minimum contrust threshold for a stable keypoint
            - R_TH: the maximum threshold for the ratio of principal curvatures, if the ratio exceeds this threshold that indicates that the 
                keypoint is an edge, and since that edges can't be keypoints the keypoint is discarded.
            - num_octaves: the number of octaves in the pyramid 
            - intervals: the order of the image that is blurred with 2 sigma in the octave

        Returns:
            - gaussian_images_pyramid: A numpy array of arrays in which the generated octaves are stored.
            - DOG_pyramid: A numpy array of arrays in which the difference of gaussians of all octaves are stored.
            - keypoints: A numpy array of arrays in which the keypoints of all octaves are stored.
    """
    # 1.6 is the value that is published that it shows maximum efficiency
    sigma= 1.6
    # generate the kernels required for generating images per octave 
    gaussian_kernels= generateGaussianKernels(sigma, intervals)
    # The pyramid of different octaves 
    gaussian_images_pyramid = []
    # the pyramid of the difference of gaussians of different octaves 
    DOG_pyramid=[]
    # the keypoints for all octaves 
    keypoints=[]
    for octave_index in range(num_octaves):
        # calculate the blurred images of the current octave and the keypoints in the octave and the difference of gaussians which is the subtraction 
        # of each two adjacent gaussian filtered images in the octave
        gaussian_images_in_octave, DOG_octave,keypoints_per_octave= generate_gaussian_images_in_octave(img, gaussian_kernels, t_c, R_TH)
        # append the current octave to the pyramid of octaves
        gaussian_images_pyramid.append(gaussian_images_in_octave)
        # append the difference of gaussians images of the current octave to the different of gaussians pyramid
        DOG_pyramid.append(DOG_octave)
        # append the keypoints of the current octave to the keypoints array
        keypoints.append(keypoints_per_octave)
        # Downsample the image that is blurred by 2 sigma to be the base image for the next octave
        img = gaussian_images_in_octave[-3][::2, ::2]
    return gaussian_images_pyramid, DOG_pyramid, keypoints

def generate_gaussian_images_in_octave(image, gaussian_kernels, t_c, R_th):
    """
        Description:
            - Generates the octave's increasingly blurred images.

        Args:
            - image: the base image for the octave.
            - gaussian_kernels: A numpy array of arrays in which the generated Gaussian kernels are stored.
            - t_c: the minimum contrust threshold for a stable keypoint
            - R_TH: the maximum threshold for the ratio of principal curvatures, if the ratio exceeds this threshold that indicates that the 
                keypoint is an edge, and since that edges can't be keypoints the keypoint is discarded.

        Returns:
            - gaussian_images_in_octave: A numpy array of arrays in which the generated blurred octave images are stored.
            - np.concatenate([o[:,:,np.newaxis] for o in DOG_octave], axis=2): 3D array representing the Difference of gaussians stacked together along the depth dimension.
            - keypoints: A numpy array of arrays in which the keypoints of the current octave are stored.
    """
    # pad the image to perserve its size in the octave
    padded_image= pad_image(image)
    # array of all the increasingly blurred images per octave 
    gaussian_images_in_octave = []  
    image = convolve(padded_image, gaussian_kernels[0])
    # append the first gaussian filtered image to the octave images
    gaussian_images_in_octave.append(image)
    # array to store the difference of each two adjacent gaussian filtered image in the octave 
    DOG_octave=[]
    # octave keypoints
    keypoints=[]
    for gaussian_kernel in gaussian_kernels[1:]:
        # convolve the gaussian kernels with the octave base padded image 
        image = convolve(padded_image, gaussian_kernel)
        gaussian_images_in_octave.append(image)
        # subtract each two adjacent images and add the result to the difference of gaussians of the octave 
        DOG_octave.append(gaussian_images_in_octave[-1]- gaussian_images_in_octave[-2])
        if len(DOG_octave)> 2:
            # from each three difference of gaussians images, detect possible keypoints through extrema detection then applying keypoints localization 
            # and filtering to discarde unstable keypoints 
            keypoints.extend(get_keypoints(DOG_octave[-3:], len(DOG_octave)-2, t_c, R_th, np.concatenate([o[:,:,np.newaxis] for o in DOG_octave], axis=2)))
    return gaussian_images_in_octave, np.concatenate([o[:,:,np.newaxis] for o in DOG_octave], axis=2), keypoints

def get_keypoints(DOG_octave, k, t_c, R_th, DoG_full_array):
    """
        Description:
            - from each three difference of gaussians images, detect possible keypoints through extrema detection which is done by comparing the middle pixel with
                its eight neighbors in the middle image and nine neighbors in the scale above and below it, then applying keypoints localization 
                and filtering to discarde unstable keypoints
        Args:
            - DOG_octave: the last three difference of gaussians calculated.
            - k: the depth of the center pixel in the difference of gaussians array.
            - t_c: the minimum contrust threshold for a stable keypoint
            - R_TH: the maximum threshold for the ratio of principal curvatures, if the ratio exceeds this threshold that indicates that the 
                keypoint is an edge, and since that edges can't be keypoints the keypoint is discarded.

        Returns:
            - keypoints: A numpy array of arrays in which the keypoints of the current octave are stored.
    """
    keypoints = [] 
    # stack the last three difference of gaussians along the depth dimention
    DoG= np.concatenate([o[:,:,np.newaxis] for o in DOG_octave], axis=2)
    # loop over the middle image and form a 3*3*3 patch 
    for i in range(1, DoG.shape[0] - 11): 
        for j in range(1, DoG.shape[1] - 11): 
            # form a (3*3*3)patch: 3 rows from i-1 to i+1, three columns from j-1 to j+1 and the depth of DoG stack is already three
            patch = DoG[i-1:i+2, j-1:j+2,:] 
            # flatten the 27 values of the patch, get the index of the maximum and minimum values of the flattened array, since the total length is 27
            # then the middle pixel index is 13 so if the returned index is 13 then the center pixel is an extrema 
            if (np.argmax(patch) == 13 or np.argmin(patch) == 13):
                # localize the detected keypoint 
                offset, J, H, x, y, s =localize_keypoint(DoG_full_array, j, i, k ) 
                # calculate its contrast 
                contrast = DoG[y,x,s] + .5*J.dot(offset) 
                # if the contrast is below the threshold move to the next patch
                if abs(contrast) < t_c: continue 
                # The eigenvalues of the Hessian matrix are used to check the ratio of principal curvatures. 
                w, v = np.linalg.eig(H)
                r = w[1]/w[0] 
                R = (r+1)**2 / r 
                # If this ratio is above a certain threshold then the keypoint is an edge therefore skip it and move to the next patch
                if R > R_th: continue 
                # add the final offset to the location of the keypoint to get the interpolated estimate for the location of the keypoint.
                kp = np.array([x, y, s]) + offset
                # append the keypoint location to the keypoints of the octave 
                keypoints.append(kp)
    return np.array(keypoints)

def localize_keypoint(D, x, y, s): 
    """
        Description:
            - refining the detected keypoints to sub-pixel accuracy. This is done by fitting a 3D quadratic function 
                to the nearby data to determine the interpolated location of the maximum, In SIFT the second-order Taylor expansion of the DoG octave is used 

        Args:
            - D: last three difference of gaussians stacked along the depth dimention
            - x: the x coordinate of the keypoint.
            - y: the y coordinate of the keypoint 
            - s: the depth of the keypoint.

        Returns:
            - offset: the final offset that should be added to the location of the keypoint to get the interpolated estimate for the location of the keypoint
            - J: the first derivatives of D, These derivatives represent the rate of change of difference of gaussians intensity in each direction.
            - HD[:2,:2]: the second derivatives (Hessian matrix) of the image intensity at the specified point. 
                The Hessian matrix represents the local curvature or second-order rate of change of image intensity.
            - x: the x coordinate of the keypoint after further localization.
            - y: the y coordinate of the keypoint after further localization.
            - s: the depth of the keypoint after further localization..
    """
    # convert D to larger data type (float) to avoid overflow
    D = D.astype(np.float64)
    #computes the first derivatives (gradient) of the image intensity along the x, y, and scale dimensions at the specified point (x, y, s). 
    dx = (D[y,x+1,s]-D[y,x-1,s])/2. 
    dy = (D[y+1,x,s]-D[y-1,x,s])/2. 
    ds = (D[y,x,s+1]-D[y,x,s-1])/2. 
    # computes the second derivatives (Hessian matrix) of the image intensity at the keypoint. 
    dxx = D[y,x+1,s]-2*D[y,x,s]+D[y,x-1,s] 
    dxy = ((D[y+1,x+1,s]-D[y+1,x-1,s]) - (D[y-1,x+1,s]-D[y-1,x-1,s]))/4
    dxs = ((D[y,x+1,s+1]-D[y,x-1,s+1]) - (D[y,x+1,s-1]-D[y,x-1,s-1]))/4 
    dyy = D[y+1,x,s]-2*D[y,x,s]+D[y-1,x,s] 
    dys = ((D[y+1,x,s+1]-D[y-1,x,s+1]) - (D[y+1,x,s-1]-D[y-1,x,s-1]))/4
    dss = D[y,x,s+1]-2*D[y,x,s]+D[y,x,s-1] 
    J = np.array([dx, dy, ds]) 
    # the second derivatives (Hessian matrix) of the image intensity at the specified point. 
    HD = np.array([ [dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]) 
    # the final offset that should be added to the location of the keypoint to get the interpolated estimate for the location of the keypoint
    offset = -np.linalg.inv(HD).dot(J)
    return offset, J, HD[:2,:2], x, y, s

image= cv2.imread(r'dog.jpg', cv2.IMREAD_GRAYSCALE)

pyramid, DoG, keypoints= generate_octaves_pyramid(image, 0.03, (10+1)**2 /10)

print(len(keypoints[1]))
print(len(keypoints[0]))
print(len(keypoints[2]))
print(len(keypoints[3]))

def visualize_pyramid(pyramid):
    fig, axes = plt.subplots(nrows=len(pyramid), ncols=len(pyramid[0]), figsize=(12, 12))

    for i in range(len(pyramid)):
        for j in range(len(pyramid[i])):
            axes[i, j].imshow(pyramid[i][j], cmap='gray')
            axes[i, j].set_title(f'Octave {i}, Image {j}')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()
    
def visualize_DOC_for_octave(DOG):
    # 2 for second octave if you need to visualize different octave update it to 0 or 1 or 3 as you like
    array= np.array(DOG[2]) 
    # 2 for second octave if you need to visualize different octave update it to 0 or 1 or 3 as you like
    for i in range(array.shape[2]):  
        plt.imshow(array[:, :, i], cmap='gray')
        plt.title(f'Image {i}')
        plt.axis('off')
        plt.show()
        
def visualize_keypoints(pyramid, keypoints):
    fig, axes = plt.subplots(nrows=len(pyramid), ncols=len(pyramid[0]), figsize=(12, 12))

    for i in range(len(pyramid)):
        for j in range(len(pyramid[i])):
            axes[i, j].imshow(pyramid[i][j], cmap='gray')
            axes[i, j].set_title(f'Octave {i}, Image {j}')
            axes[i, j].axis('off')
            for kp in keypoints[i]:
                x = kp[0]
                y = kp[1]
                circle = Circle((x, y), radius=2, color='r', fill=True)
                axes[i, j].add_patch(circle)  

    plt.tight_layout()  
    plt.show()
visualize_pyramid(pyramid)
# visualize_DOC_for_octave(DoG)
# visualize_keypoints(pyramid, keypoints)  # take care: takes alot of time since it loops through the four octaves and their five filtered images and draw the keypoints on each
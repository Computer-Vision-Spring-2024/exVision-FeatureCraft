import numpy as np
from scipy.signal import convolve2d
import cv2
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
from matplotlib.patches import Circle 
from PIL import Image
from math import sin, cos



def gaussian_filter_kernel(sigma, kernel_size = None):
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
    gaussian_kernels= []
    scale_level= sigma 
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

def generate_octaves_pyramid(img,  num_octaves=4, s_value= 2,sigma = 1.6, contrast_th = 0.03, ratio_th = 10):
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
    gaussian_kernels = generateGaussianKernels(sigma, s_value) # intervals == s  ( s + 3 )
    # The pyramid of different octaves 
    gaussian_images_pyramid = []
    # the pyramid of the difference of gaussians of different octaves 
    DOG_pyramid=[]
    # the keypoints for all octaves 
    keypoints=[]
    for octave_index in range(num_octaves):
        # calculate the blurred images of the current octave and the keypoints in the octave and the difference of gaussians which is the subtraction 
        # of each two adjacent gaussian filtered images in the octave
        gaussian_images_in_octave, DOG_octave , keypoints_per_octave= generate_gaussian_images_in_octave(img, gaussian_kernels, contrast_th, ratio_th, octave_index)
        # append the current octave to the pyramid of octaves
        gaussian_images_pyramid.append(gaussian_images_in_octave)
        # append the difference of gaussians images of the current octave to the different of gaussians pyramid
        DOG_pyramid.append(DOG_octave)
        # append the keypoints of the current octave to the keypoints array
        keypoints.append(keypoints_per_octave)
        # Downsample the image that is blurred by 2 sigma to be the base image for the next octave
        img = gaussian_images_in_octave[-3][::2, ::2]
    return gaussian_images_pyramid, DOG_pyramid, keypoints

def generate_gaussian_images_in_octave(image, gaussian_kernels, contrast_th, ratio_th, octave_index):
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

    DOG_octave=[]
    # octave keypoints
    keypoints=[]
    for gaussian_kernel in gaussian_kernels[1:]:
        # convolve the gaussian kernels with the octave base padded image 
        blurred_image = convolve2d(image, gaussian_kernel , "same", "symm")
        gaussian_images_in_octave.append(blurred_image)
        # subtract each two adjacent images and add the result to the difference of gaussians of the octave 
        DOG_octave.append(gaussian_images_in_octave[-1]- gaussian_images_in_octave[-2])
        if len(DOG_octave)> 2:
            # from each three difference of gaussians images, detect possible keypoints through extrema detection then applying keypoints localization 
            # and filtering to discarde unstable keypoints 
            keypoints.extend(get_keypoints(DOG_octave[-3:], len(DOG_octave)-2, contrast_th, ratio_th, np.concatenate([o[:,:,np.newaxis] for o in DOG_octave], axis=2)))
    return gaussian_images_in_octave, DOG_octave, keypoints

def get_keypoints(DOG_octave, k, contrast_th , ratio_th, DoG_full_array): 
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
    DoG= np.concatenate([o[:,:,np.newaxis] for o in DOG_octave], axis=2) # 2d -- > 3d 
    # loop over the middle image and form a 3*3*3 patch 
    for i in range(1, DoG.shape[0] - 2): 
        for j in range(1, DoG.shape[1] - 2): 
            # form a (3*3*3)patch: 3 rows from i-1 to i+1, three columns from j-1 to j+1 and the depth of DoG stack is already three
            patch = DoG[i-1:i+2, j-1:j+2,:] 
            # flatten the 27 values of the patch, get the index of the maximum and minimum values of the flattened array, since the total length is 27
            # then the middle pixel index is 13 so if the returned index is 13 then the center pixel is an extrema 
            if (np.argmax(patch) == 13 or np.argmin(patch) == 13):
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
                kp = np.array([i,j,k])
                keypoints.append(kp)
    return np.array(keypoints)

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
    H = np.array([ [dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]) 
    # the final offset that should be added to the location of the keypoint to get the interpolated estimate for the location of the keypoint
    offset = -np.linalg.inv(H).dot(J) # ((3 x 3) . 3 x 1)    
    return offset, J, H[:2,:2], x, y, s 



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
    fig, axes = plt.subplots(nrows=len(DOG), ncols=len(DOG[0]), figsize=(12, 12))

    for i in range(len(DOG)):
        for j in range(len(DOG[i])):
            axes[i, j].imshow(DOG[i][j], cmap='gray')
            axes[i, j].set_title(f'Octave {i}, Image {j}')
            axes[i, j].axis('off')

    plt.tight_layout()
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

def sift_resize(img, ratio = None):
    ratio = ratio if ratio is not None else np.sqrt((1024*1024) / np.prod(img.shape[:2]))
    newshape = list(map( lambda d : int(round(d*ratio)), img.shape[:2])) 
    img = resize( img, newshape , anti_aliasing = True )
    return img,ratio

def convert_to_grayscale(image):
    if len(image.shape) == 3: 
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    return image

def represent_keypoints(keypoints, DoG):
    keypoints_as_images = list()
    for octave_ind ,kp_per_octave in enumerate(keypoints):
        keypoints_per_octave = list()
        for dog_idx in range(1, len(DoG)-1):
            keypoints_per_sigma =  np.full( DoG[octave_ind][0].shape, False, dtype = bool) # create bool 2d array of same size as 
            for kp in kp_per_octave:
                if kp[2] == dog_idx: 
                    keypoints_per_sigma[kp[0],kp[1]] = True
            keypoints_per_octave.append(keypoints_per_sigma)
        keypoints_as_images.append(keypoints_per_octave)
    return keypoints_as_images



def sift_gradient(img):
    dx = np.array([-1,0,1]).reshape((1,3)); dy = dx.T
    gx = convolve2d( img , dx , boundary='symm', mode='same' )
    gy = convolve2d( img , dy , boundary='symm', mode='same' )
    magnitude = np.sqrt( gx * gx + gy * gy )
    direction = np.rad2deg( np.arctan2( gy , gx )) % 360
    return gx, gy, magnitude, direction

def padded_slice(img, sl):
    output_shape = np.asarray(np.shape(img))
    output_shape[0] = sl[1] - sl[0]
    output_shape[1] = sl[3] - sl[2]
    src = [max(sl[0], 0),
           min(sl[1], img.shape[0]),
           max(sl[2], 0),
           min(sl[3], img.shape[1])]
    dst = [src[0] - sl[0], src[1] - sl[0],
           src[2] - sl[2], src[3] - sl[2]]
    output = np.zeros(output_shape, dtype=img.dtype)
    output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
    return output

def dog_keypoints_orientations( img_gaussians , keypoints , sigma_base ,num_bins = 36, s = 2):
    kps = []
    for octave_idx in range(len(img_gaussians)): # iterate over the ocataves
        img_octave_gaussians = img_gaussians[octave_idx]
        octave_keypoints = keypoints[octave_idx]
        for idx, scale_keypoints in enumerate(octave_keypoints):
            scale_idx = idx + 1 ## idx+1 to be replaced by quadratic localization
            gaussian_img = img_octave_gaussians[scale_idx] 
            sigma = 1.5 * sigma_base * ( 2 ** octave_idx ) * ( (2**(1/s)) ** (scale_idx))

            kernel = gaussian_filter_kernel(sigma)
            radius = int(round(sigma * 2)) # 2 x std == 95 % 
            gx,gy,magnitude,direction = sift_gradient(gaussian_img)
            direction_idx = np.round( direction * num_bins / 360 ).astype(int)          
            
            for i,j in map( tuple , np.argwhere( scale_keypoints ).tolist() ):
                window = [i-radius, i+radius+1, j-radius, j+radius+1]
                mag_win = padded_slice( magnitude , window )
                dir_idx = padded_slice( direction_idx, window )
                weight = mag_win * kernel 
                hist = np.zeros(num_bins, dtype=np.float32)
                
                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum( weight[ dir_idx == bin_idx ] )
            
                for bin_idx in np.argwhere( hist >= 0.8 * hist.max() ).tolist():
                    angle = (bin_idx[0]+0.5) * (360./num_bins) % 360
                    kps.append( (i,j,octave_idx,scale_idx,angle))
    return kps


def rotated_subimage(image, center, theta, width, height):
    theta *= 3.14159 / 180 # convert to rad
    
    
    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)

def get_gaussian_mask(sigma,filter_size):
    if sigma > 0: 
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (filter_size-1)/2)**2 + (y - (filter_size-1)/2)**2) / (2*sigma**2)), (filter_size, filter_size))
        return kernel / np.sum(kernel)
    else:
        raise ValueError("Invalid value of Sigma")

def extract_sift_descriptors( img_gaussians, keypoints, base_sigma,num_bins = 8, s = 2):
    descriptors = []; points = [];  data = {} # 
    for (i,j,oct_idx,scale_idx, orientation) in keypoints:

        if 'index' not in data or data['index'] != (oct_idx,scale_idx):
            data['index'] = (oct_idx,scale_idx)
            gaussian_img = img_gaussians[oct_idx][ scale_idx ] 
            sigma = 1.5 * base_sigma * ( 2 ** oct_idx ) * ( (2**(1/s)) ** (scale_idx))
            data['kernel'] = get_gaussian_mask(sigma = sigma, filter_size = 16)           

            gx,gy,magnitude,direction = sift_gradient(gaussian_img)
            data['magnitude'] = magnitude
            data['direction'] = direction

        window_mag = rotated_subimage(data['magnitude'],(j,i), orientation, 16,16)
        window_mag = window_mag * data['kernel']
        window_dir = rotated_subimage(data['direction'],(j,i), orientation, 16,16)
        window_dir = (((window_dir - orientation) % 360) * num_bins / 360.).astype(int)

        features = []
        for sub_i in range(4):
            for sub_j in range(4):
                sub_weights = window_mag[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                sub_dir_idx = window_dir[sub_i*4:(sub_i+1)*4, sub_j*4:(sub_j+1)*4]
                hist = np.zeros(num_bins, dtype=np.float32)
                for bin_idx in range(num_bins):
                    hist[bin_idx] = np.sum( sub_weights[ sub_dir_idx == bin_idx ] )
                features.extend( hist.tolist())
        features = np.array(features) 
        features /= (np.linalg.norm(features))
        np.clip( features , np.finfo(np.float16).eps , 0.2 , out = features )
        assert features.shape[0] == 128, "features missing!"
        features /= (np.linalg.norm(features))
        descriptors.append(features)
        points.append( (i ,j , oct_idx, scale_idx, orientation))
    return points , descriptors


def computeKeypointsAndDescriptors(image, n_octaves, s_value, sigma_base, constract_th, r_ratio):
    grayscaled_image = convert_to_grayscale(image)    
    base_image = rescale( grayscaled_image, 2, anti_aliasing=False) 
    pyramid, DoG, keypoints = generate_octaves_pyramid(base_image, n_octaves, s_value, sigma_base, constract_th, r_ratio ) 
    keypoints = represent_keypoints(keypoints, DoG) 
    keypoints_ijso = dog_keypoints_orientations( pyramid , keypoints, sigma_base , 36, s_value )  # ( i ,j , oct_idx, scale_idx, orientation)
    points,descriptors = extract_sift_descriptors(pyramid , keypoints_ijso, sigma_base, 8, s_value)
    return points,descriptors

def kp_list_2_opencv_kp_list(kp_list):

    opencv_kp_list = []
    for kp in kp_list:
        opencv_kp = cv2.KeyPoint(x=kp[1] * (2**(kp[2]-1)),
                                 y=kp[0] * (2**(kp[2]-1)),
                                 size=kp[3],
                                 angle=kp[4],
                                 )
        opencv_kp_list += [opencv_kp]

    return opencv_kp_list

def draw_matches(img1, kp1, img2, kp2, matches, colors,count): 

    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 3
    thickness = 1
    for idx in range(min(count,len(matches))):
        m = matches[idx]
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, colors[idx], thickness)
        cv2.circle(new_img, end1, r, colors[idx], thickness)
        cv2.circle(new_img, end2, r, colors[idx], thickness)
    


def match( img_a, pts_a, desc_a, img_b, pts_b, desc_b, tuning_distance = 0.3 ):
    img_a, img_b = tuple(map( lambda i: np.uint8(i*255), [img_a,img_b] ))
    
    desc_a = np.array( desc_a , dtype = np.float32 )
    desc_b = np.array( desc_b , dtype = np.float32 )

    pts_a = kp_list_2_opencv_kp_list(pts_a)
    pts_b = kp_list_2_opencv_kp_list(pts_b)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_a,desc_b,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < tuning_distance * n.distance:
            good.append(m)

    img_match = np.empty((max(img_a.shape[0], img_b.shape[0]), img_a.shape[1] + img_b.shape[1], 3), dtype=np.uint8)

    cv2.drawMatches(img_a,pts_a,img_b,pts_b,good, outImg = img_match,
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_match



n_octaves = 4 # entry box 
s_value = 2 # entry box 
sigma_base = 1.6 # entry box float
r_ratio = 10 # entry box float
# -------------
contrast_th = 0.03 #  slider 
tuning_factor = 0.3 # slider 

main_image = np.array(Image.open("img.jpg"))
template = np.array(Image.open("02.jpg"))

def apply_sift(main_image, template ,n_octaves, s_value, sigma_base, contrast_th, r_ratio, tuning_factor): 

    main_image, ratio = sift_resize(main_image)
    template, _ = sift_resize(template, ratio)

    img_kp, img_des = computeKeypointsAndDescriptors(main_image, n_octaves, s_value, sigma_base, contrast_th, r_ratio)
    template_kp, template_des  = computeKeypointsAndDescriptors(template, n_octaves, s_value, sigma_base, contrast_th, r_ratio)

    img_match = match(main_image, img_kp, img_des,template , template_kp, template_des, tuning_factor)
    return  img_match

img_match = apply_sift(main_image, template ,n_octaves, s_value, sigma_base, contrast_th, r_ratio, tuning_factor)

plt.figure(figsize=(20,20))
plt.imshow(img_match)
plt.show() 
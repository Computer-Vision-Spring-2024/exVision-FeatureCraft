import cv2
import numpy as np

# Global variables to store seed point and threshold
seed_point = (-1, -1)
threshold = 10

def region_growing(image, seed_point, threshold):
    # Create a binary mask to keep track of visited pixels
    rows, cols = image.shape[:2]
    visited = np.zeros((rows, cols), dtype=np.uint8)
    
    # Create a queue to store the pixels to be visited
    queue = []
    
    # Get the intensity value of the seed point
    seed_value = image[seed_point[1], seed_point[0]]
    
    # Add the seed point to the queue
    queue.append(seed_point)
    
    # Create a segmented image with the same size as the original image
    segmented_image = np.zeros_like(image)
    
    # Define the 4-connectivity kernel
    connectivity_kernel = np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]], dtype=np.uint8)
    
    # Region growing process
    while len(queue) > 0:
        # Get the current pixel from the queue
        current_point = queue.pop(0)
        x, y = current_point
        
        # Check if the current pixel is within the image boundaries and not visited yet
        if (0 <= x < rows) and (0 <= y < cols) and (visited[y, x] == 0):
            # Mark the current pixel as visited
            visited[y, x] = 1
            
            # Check if the intensity difference between the current pixel and the seed point is below the threshold
            if abs(int(image[y, x]) - int(seed_value)) < threshold:
                # Add the current pixel to the segmented region
                segmented_image[y, x] = 255
                
                # Add neighboring pixels to the queue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            queue.append((x + dx, y + dy))
                            
    return segmented_image

def mouse_callback(event, x, y, flags, param):
    global seed_point
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_point = (x, y)
        print("Seed point set at:", seed_point)

# Load the image
image = cv2.imread("C:/Users/Kareem/Desktop/ImageAlchemy/App/Resources/ImagesLibrary/Medical/Brain2.pgm", cv2.IMREAD_GRAYSCALE)

# Create a window and set mouse callback function
cv2.namedWindow("Select Seed Point")
cv2.setMouseCallback("Select Seed Point", mouse_callback)

# Wait for the user to select the seed point
while seed_point == (-1, -1):
    cv2.imshow("Select Seed Point", image)
    cv2.waitKey(1)

# Apply region growing
segmented_image = region_growing(image, seed_point, threshold)

# Display the original and segmented images
cv2.imshow("Original Image", image)
cv2.imshow("Segmented Image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

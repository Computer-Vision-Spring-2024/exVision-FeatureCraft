# exVision: FeatureCraft

## Overview
FeatureCraft is a desktop application that leverages classical feature engineering techniques to perform template matching, with a specific focus on extracting invariant features for template matching. The application incorporates the Scale-Invariant Feature Transform (SIFT) algorithm, various corner detection methods (e.g., the Harris Corner Detector), and a specialized technique called lambda-minus, all aimed at identifying robust features across different conditions, such as changes in scale, rotation, and illumination.

## Feature Engineering Techniques

### Corners Detection (Harris & Lambda-Minus)
Corners are important features in image processing because they represent points where two edges meet, often providing distinctive, robust, rotation-invariant features for tasks like object detection and image matching. The Harris Corner Detection algorithm is a popular method for detecting such corners.
1. **Harris Corner Detection:** This algorithm identifies corners by analyzing the local intensity gradients within an image. It computes the Harris matrix (also known as the structure tensor) at each pixel, based on image derivatives, to capture changes in intensity. The corner score, known as the Harris response, is calculated using an equation that considers the eigenvalues of this matrix. A high response value indicates potential corner points.
<p align="center">
  <img src="README-Assets\images\harris_corners.png" alt="corner detection" width="500"/>
</p>

2. **Lambda-Minus:** The method uses the eigenvalues (λ1, λ2) of the Harris matrix to assess feature points:
    - If both eigenvalues are large, the point is likely a corner.
    - If one eigenvalue is large and the other is small, the point lies on an edge.
    - If both eigenvalues are small, the point is in a flat region.

<p align="center">
  <img src="README-Assets\images\harris-demo.jpg" alt="lambda-minus" width="300"/>
</p>

### Scale-and-Rotation-Invariant Blob Detection (SIFT) 
The Scale-Invariant Feature Transform (SIFT) is used to detect and describe local features in images. It is robust to changes in scale, rotation, and illumination, making it ideal for matching key points across different images of the same object or scene. The SIFT algorithm works by:

1. **Keypoint Detection:** Identifying potential points of interest, or "blobs," that remain stable under scale changes. This is done by detecting extrema in scale-space.

2. **Keypoint Localization:** Refining the location and scale of each keypoint to improve accuracy.

3. **Orientation Assignment:** Assigning a dominant orientation to each keypoint to ensure invariance to image rotation.

4. **Keypoint Descriptor:** Generating a distinctive descriptor for each keypoint based on the local image gradient around it. This allows for reliable matching between keypoints in different images.

## Template Matching using SIFT 

By comparing the keypoint descirptors of the main image and template, we can map locate some objects in the main image. 



<p align="center">
  <img src="README-Assets\images\template_matching.png" alt="rotation invariance" width="400"/>
</p>

**For a more in-depth understanding of each each feature engineering technique, please refer to the [attached notebooks and python scripts](implementation_without_ui) as well as [the project documentation](README-Assets/FeatureCraft-Documentation.pdf).**

## Getting Started

To be able to use our app, you can simply follow these steps:
1. Install Python3 on your device. You can download it from <a href="https://www.python.org/downloads/">Here</a>.
2. Install the required packages by the following command.
```
pip install -r requirements.txt
```
3. Run the file with the name "FeatureCraft_Backend.py" in the [app folder](FeatureCraft-PyQt-App).

## Acknowledgments

- Refer to [this organization's README](https://github.com/Computer-Vision-Spring-2024#acknowledgements) for more details about contributors and supervisors. 
- "Corner detection." (n.d.). In *Wikipedia*. Retrieved from [Wiki](https://en.wikipedia.org/wiki/Corner_detection)

## References 

- Lowe, D. G. (2004). *Distinctive image features from scale-invariant keypoints*. Computer Science Department, University of British Columbia.

- Otero, I. R., & Delbracio, M. (2014). *Anatomy of the SIFT method*. Image Processing On Line, 4, 370–396. 



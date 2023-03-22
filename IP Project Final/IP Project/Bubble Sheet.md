
# Module (1): Bubble Sheet

## Preprocessing

    The preprocessing step is the first step in the image processing pipeline. It is used to remove noise and to enhance the image. The preprocessing step is also used to prepare the image for the segmentation step. 
    
    Before applying any preprocessing however, we must first crop the image. The image is cropped to remove the background and to only include the image of the paper. 
    
    This is done by first finding all the contours in the image. It is assumed that the contour with the largest area is the contour of the paper. And the contour with the second largest area is the contour of the bubble sheet.

    Once the contours are found the corners are extracted and image is warped using OpenCv's getPerspectiveTransform function.

    After the image is cropped, the preprocessing step is applied. Using adaptive local thresholding, the image is binarized and to remove noise a median filter is then applied.

## Input and Output Example
<p float="left">
    <img src="Bubble_sheet\4\IMG_2039.jpg" width=300>
    <img src="Bubble_sheet\Outputs\Warped_Thresholded.jpg" width=300>
</p>


## Segmentation and Analysis

    Once the image is preprocessed, the segmentation step is applied. The segmentation step is used to segment the image into individual bubbles. 

    It can be summarized as 
        1. Apply circular Hough transform to find circles in the image.
        2. Sort the circles from top left to bottom right and determine which circles belong to the same row.            
            - This is done by first finding the top left and top right bubbles in the unsorted list of circles.
            - A line is then drawn between the top left and top right bubbles.
            - Circles intersecting with the line are then added to the row.
            - The row is then sorted from left to right.
            - Repeat the process until all the circles are sorted into rows.
        3. Rows are then further categorized into ID rows and answer rows.
            - ID rows are rows before a large vertical gap. between the rows.
            - Answer rows are all the rows after the ID rows.
        4. The number of columns in the answer rows are determined by the large horizontal gap between the bubbles.
Great thanks to user S. Vogt on [StackOverflow](https://stackoverflow.com/questions/29630052/ordering-coordinates-from-top-left-to-bottom-right) for the main idea behind sorting the circles.
  

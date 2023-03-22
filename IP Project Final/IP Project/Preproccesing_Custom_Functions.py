from skimage.measure import find_contours
from skimage.color import rgb2gray
from skimage.morphology.binary import binary_dilation
from skimage.filters._median import median
from skimage.filters.thresholding import threshold_local
import skimage.io as io
import numpy as np
import cv2


def Get_Contour_Area(Contour):
    '''This function takes a contour as input and returns the area of the contour'''
    
    
    #Extract the corners of the current contour in the loop
    Top_Left = sorted(Contour, key= lambda x: x[0]+x[1])[0]
    Top_Right = sorted(Contour, key= lambda x: x[0]-x[1])[-1]
    Bottom_Left = sorted(Contour, key= lambda x: x[0]-x[1])[0]
    Bottom_Right = sorted(Contour, key= lambda x: x[0]+x[1])[-1]

    # Contour area is the product of the length and width of the contour
    Length = np.sqrt((Top_Left[0]-Top_Right[0])**2 + (Top_Left[1]-Top_Right[1])**2)
    Width = np.sqrt((Top_Left[0]-Bottom_Left[0])**2 + (Top_Left[1]-Bottom_Left[1])**2)
    
    
    Contour_Area = Length*Width
    
    return Contour_Area
    
    
def Get_Nth_Largest_Contour(gray_img, N, mode='Normal') -> list:
    '''This function takes a grayscale image and returns the Nth largest contour in that image based on the area of the contour'''
    # Find the contours in the image
    x,y=gray_img.shape
    threshold= ((gray_img.max()+gray_img.min())/2)*1.2
    
    Contours=find_contours(gray_img[5:x-5,5:y-5], threshold)
    
    while len(Contours)<3:
        threshold*=1.05
        Contours=find_contours(gray_img[5:x-5,5:y-5], threshold)
   
    # Sort the contours by area
    Contours = sorted(Contours, key=Get_Contour_Area, reverse=True)
    
    # If the contour being searched for is the paper or border, it shouldn't consume less than 20% of the image unless something is wrong.
    if mode=='border' or mode== 'paper':
        while Get_Contour_Area(Contours[N]) < 0.3 * gray_img.shape[0] * gray_img.shape[1]:
            threshold*=1.1
            Contours=find_contours(gray_img,threshold)
            Contours = sorted(Contours, key=Get_Contour_Area, reverse=True)
            
            

    return Contours[N-1]


def Mask_From_Contour(gray_img, Contour) -> np.ndarray:
    '''This function takes a grayscale image and a contour as input and returns a mask of the contour'''
    # Create a mask of the contour
    mask = np.zeros_like(gray_img)
    # Switch the x and y coordinates of the contour since the drawContours function expects the coordinates in the opposite order
    copy=Contour.copy()
    copy[:,[0, 1]] = copy[:,[1, 0]]
    cv2.drawContours(mask, [copy.astype(np.int32)], 0, 1, -1)
    return mask


def Get_Corners(contour) -> np.ndarray:
    '''This function takes a contour as input and returns the corners of the mask, sorted in the order of top left, top right, bottom left, bottom right'''

    Top_Left = sorted(contour, key= lambda x: x[0]+x[1])[0]
    Top_Right = sorted(contour, key= lambda x: x[0]-x[1])[-1]
    Bottom_Left = sorted(contour, key= lambda x: x[0]-x[1])[0]
    Bottom_Right = sorted(contour, key= lambda x: x[0]+x[1])[-1]

    Corners=np.array([Top_Left,Top_Right,Bottom_Left,Bottom_Right])
    
    return Corners

def Get_Perspective_Transform(img, Corners,mode='border') -> np.ndarray:
    size_multiplier = 5
    if mode == 'border' or mode == 'paper':
        width,height = 210*size_multiplier, 297*size_multiplier # A4 paper aspect ratio 210mm x 297mm
    if mode == 'name':
        height,width = 210, 297 # Name in the box aspect ratio 210mm x 297mm
        
    pts1 = np.float32([Corners[0], Corners[1], Corners[2], Corners[3]]) # The 4 corners of the contour
    pts2= np.float32([[0,0],[width,0],[0,height],[width,height]]) # The corners of the final image
    matrix = cv2.getPerspectiveTransform(pts1,pts2) # The transformation matrix
    result = cv2.warpPerspective(img,matrix,(width,height)) # The warped image
    return result

def Extract_Sheet(path:str,mode:str='border') -> np.ndarray:
    '''This function takes the path of the image as input and returns the bubble sheet from the image
        mode can be 'border' or 'paper' or 'name
        'border' mode returns the bubble sheet with the inner border
        'paper' mode returns the bubble sheet paper whole
        'name' mode returns the name in the box, input image should be the warped image'''
  
    # Read the image
    img_rgb = io.imread(path)
    # Convert the image to grayscale
    img_gray = rgb2gray(img_rgb)
    # Get the N largest contour, paper is the largest contour, inner border is the second largest contour
    if mode=='paper' or mode=='name': N=1 # Name should be the largest contour in the warped image
    if mode=='border': N=2
    
    contour = Get_Nth_Largest_Contour(img_gray, N,mode) # First largest contour is the paper, second largest contour is the inner border
    # Get the mask of the inner border, used to extract the corners of the inner border, as it is much easier to extract the corners of a mask than the image.
    mask = Mask_From_Contour(img_gray, contour)
    # Get corners of the inner border
    # Harris corner detection is used to detect the corners of the inner border
    corners = Get_Corners(mask)
    # We use the perspective transform function to transform the image to a top down view of the bubble sheet
    warped_img = Get_Perspective_Transform(img_rgb, corners,mode=mode)
    # if mode != 'name':
    #     io.imsave(path[-8:-4]+'_Warped.jpg',warped_img)
    # if mode == 'name':
    #     io.imsave(path[-15:-11]+'_Name.jpg',warped_img)
    return warped_img


def Apply_Preprocessing(img, param=1, blocksize=11,offset=12, method='median', mode='Normal') -> np.ndarray:
    '''This function takes an image as input and returns the image with preprocessing applied
        param: The number of times the image is eroded.
        Blocksize: The size of the local thresholding block
        Offset: The offset of the local thresholding
        mode: 'Normal' returns the same image binarized using local thresholding.
              'No_Name' like normal but the name is removed from the image
              'Enhance' like normal but the bubbles are shaded enhanced, takes longer to process'''
    result_gray = rgb2gray(img)*255
    
    if mode=='No_Name':
        Name_Contour= Get_Nth_Largest_Contour(result_gray, 1)
        Name_Mask= Mask_From_Contour(result_gray, Name_Contour)
        result_gray*=  np.logical_not(Name_Mask)
        
    if mode=='Enhance':
        blocksize=71
        
    local_threshold = threshold_local(result_gray, blocksize, offset=offset, method=method)
    binary_local = result_gray > local_threshold
    
    if mode=='Enhance': 
        for i in range(param):
            binary_local = binary_dilation(binary_local) # Colors are flipped so we use dilation function to erode the image.

    binary_local = median(binary_local)
    return binary_local


    

    

    
    
    
    
    
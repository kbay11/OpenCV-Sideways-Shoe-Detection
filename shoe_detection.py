import cv2
import numpy as np
import matplotlib.pyplot as plt


def merge_contours(contours):
    """
    Merges overlapping contours into a single contour for each set of overlapping areas.
    This is useful for simplifying the detection results and obtaining a minimal number of bounding boxes.
    
    Args:
        contours (list of np.array): List of contours where each contour is represented as an array of points.

    Returns:
        list of np.array: The list of merged contours.
        
    Details:
        - The function first calculates the bounding rectangles for each contour.
        - It then iteratively checks for overlaps between these rectangles.
        - If two rectangles overlap, they are merged into a single rectangle.
        - This process repeats until no more merges can be made.
        - The merged rectangles are converted back to contour format before returning.
    """
    # Calculate the bounding rectangles for all contours
    rects = [cv2.boundingRect(contour) for contour in contours]
    # Boolean array to track changes during iterations
    changed = True

    while changed:
        # Keep track of whether any merges occur in this iteration
        changed = False
        # Boolean array to keep track of which contours have been merged
        merged = np.zeros(len(rects), dtype=bool)
        # Temporary list to store new set of merged rectangles
        new_rects = []
        # Iterate over all contours
        for i in range(len(rects)):
            if not merged[i]:
                x1, y1, w1, h1 = rects[i]
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x1 + w1, y1 + h1
                # Iterate over all other contours to check for overlap
                for j in range(i + 1, len(rects)):
                    if not merged[j]:
                        x2, y2, w2, h2 = rects[j]
                        # Check if rectangles overlap
                        if not (new_x2 < x2 or new_x1 > x2 + w2 or new_y2 < y2 or new_y1 > y2 + h2):
                            # Update the boundaries of the new merged rectangle to include both
                            new_x1 = min(new_x1, x2)
                            new_y1 = min(new_y1, y2)
                            new_x2 = max(new_x2, x2 + w2)
                            new_y2 = max(new_y2, y2 + h2)
                            # Mark both contours as merged
                            merged[j] = True
                            changed = True
                # Add the newly formed merged rectangle to the list
                if not merged[i]:
                    merged[i] = True
                    new_rects.append((new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1))
        # Update rects with newly formed merged rectangles
        rects = new_rects

    # Convert bounding rectangles back to contour format
    merged_contours = []
    for x, y, w, h in rects:
        contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        merged_contours.append(contour.reshape(-1, 1, 2))

    return merged_contours


def check_background(image_path, display=False):
    """
    Checks if the background of an image is primarily white to gray.

    Args:
        image_path (str): Path to the image file.
        display (bool): Whether to display the image processing steps.

    Returns:
        bool: True if the background is white to gray, False otherwise.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be loaded.")
        return False

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresholded = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

    # Calculate the average pixel intensity of the thresholded areas
    mask = thresholded == 255
    average_intensity = np.mean(gray_image[mask])

    # Define the range for white to gray
    is_white_to_gray = 180 <= average_intensity <= 255

    if display:
        # Display the original, grayscale, and thresholded images
        plt.figure(figsize=(15, 8))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(thresholded, cmap='gray')
        plt.title('Thresholded Image')
        plt.axis('off')

        plt.show()

    return is_white_to_gray

def detect_shoe_by_bounding_box(image_path, low_threshold, high_threshold, display=True, verbose=False):
    """
    Detects objects in an image using edge detection and morphological operations,
    creates bounding boxes for all merged contours, and classifies each as 'shoe' or 'not shoe'
    based on aspect ratio.

    Args:
    image_path (str): Path to the image file.
    low_threshold (int): Lower threshold for Canny edge detector.
    high_threshold (int): Upper threshold for Canny edge detector.
    display (bool): Whether to display the intermediate images.
    verbose (bool): Whether to print additional information.
    """
    # Load image and perform edge detection
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # get image aspect ratio
    height, width = img.shape

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Use dilation to close gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if verbose: print("Number of contours:", len(contours))
    # Load the original image to draw separate contours on
    separate_contours_image = cv2.imread(image_path)
    cv2.drawContours(separate_contours_image, contours, -1, (255, 0, 0), 2)

    # Merge contours that are inside each other
    merged_contours = merge_contours(contours)
    if verbose: print("Number of merged contours:", len(merged_contours))

    # Load the original image to draw on
    color_image = cv2.imread(image_path)

    # Draw bounding boxes and classify each contour
    for j,contour in enumerate(merged_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio_w2h = w / float(h) 
        aspect_ratio_h2w = h / float(w)

        bbox_area = w * h
        bbox_area_percentage = bbox_area / (width * height) * 100
        # Check if the bounding box is isolated from the image edges
        is_within_bounds = x > 0 and y > 0 and x + w < width and y + h < height

        # check if there is only one bounding box, only one bounding box is expected for a shoe
        if len(merged_contours) == 1:
            #check if the area of the bounding box is 60% or more of the image
            if verbose:
                print(f"Bounding Box Area: {bbox_area}")
                print(f"Image Area: {width * height}")
                print(f"Bounding Box Area Percentage: {bbox_area_percentage:.2f}%")  # Formatted to 2 decimal places
                print(f"Bounding Box Ratio width to height: {aspect_ratio_w2h:.2f}")  # Formatted to 2 decimal places
                print(f"Bounding Box Ratio height to width: {aspect_ratio_h2w:.2f}")  # Formatted to 2 decimal places
            # check if the bounding box is 60% or more of the image and if the bounding box is not touching the edge of the image
            if bbox_area_percentage >= 65 or not is_within_bounds:
                # Boots are usually longer than they are wide and take up more space in the image and have an inverse aspect ratio greater than 1.5
                # check if the aspect ratio is less than 0.66, 
                if aspect_ratio_h2w > 1.6:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
            else:
                # Shoes from the back are usually shorter than they are wide and take up less space in the image and have an aspect ratio less than 1.2 i.e. closer to a square
                if 0.8 < aspect_ratio_w2h < 1.2:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
       
        if verbose: 
             # Print the area of the bounding box
            print(f"Bounding Box Area: {bbox_area}")
            # Print the area of the bounding box in relation to the image
            print(f"Bounding Box Area Percentage: {bbox_area_percentage:.2f}%")


        
        # Draw the bounding rectangle
        cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 3)

    # Convert images from BGR to RGB for plotting
    # separate_contours_rgb = cv2.cvtColor(separate_contours_image, cv2.COLOR_BGR2RGB)
    merged_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)

    if True:
        # Display the images
        plt.figure(figsize=(15,8)) #15, 8)) 

        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edges')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(dilated, cmap='gray')
        plt.title('Dilated Edges')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(merged_rgb)
        plt.title('Bounding Boxes')
        plt.axis('off')

        plt.show()
    
    # Check if the color of the bounding box is green i.e. a shoe is detected
    if color == (0, 255, 0):
        return True
    else:
        return False

def detect_shoe(image_path, low_threshold, high_threshold, display_thresholding=False, display_bounding_box=True, verbose=False):
    """
    Detects objects in an image using edge detection and morphological operations,
    creates bounding boxes for all merged contours, and classifies each as 'shoe' or 'not shoe'
    based on aspect ratio.

    Args:
    image_path (str): Path to the image file.
    low_threshold (int): Lower threshold for Canny edge detector.
    high_threshold (int): Upper threshold for Canny edge detector.
    display_thresholding (bool): Whether to display the intermediate images during thresholding.
    display_bounding_box (bool): Whether to display the final bounding boxes.
    verbose (bool): Whether to print additional information.
    """

    # Check if the background is white to gray
    is_white_to_gray = check_background(image_path, display=display_thresholding)

    if is_white_to_gray:
        return detect_shoe_by_bounding_box(image_path, low_threshold, high_threshold, display_bounding_box, verbose)
    else:
        return False
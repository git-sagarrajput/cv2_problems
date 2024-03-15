import re
import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the input image to enhance its quality for rectangle detection.

    Args:
        image: A NumPy array representing the input image.

    Returns:
        A preprocessed image as a NumPy array.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Enhance contrast using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    return enhanced

def find_nested_rectangles(image):
    """
    Finds nested rectangles in a black and white image and reports their level.

    Args:
        image: A black and white image as a NumPy array.

    Returns:
        A list of tuples, where each tuple contains the coordinates of a rectangle
        and its level of nesting (0 for the outermost rectangle).
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Threshold the preprocessed image to create a binary image.
    thresh = cv2.threshold(preprocessed_image, 127, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the binary image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list to store the rectangles and their levels.
    rectangles = []

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Loop through the contours and find rectangles.
    for cnt in contours:
        # Check if the contour is a rectangle.
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Get the bounding rectangle of the contour.
            x, y, w, h = cv2.boundingRect(cnt)

            # Exclude rectangles touching the image edges
            if x == 0 or y == 0 or x + w == width or y + h == height:
                continue

            # Calculate the level of nesting for the rectangle.
            level = 0
            for other_cnt in contours:
                if np.array_equal(cnt, other_cnt):
                    continue
                other_x, other_y, other_w, other_h = cv2.boundingRect(other_cnt)
                # Check if the current rectangle completely encloses the other rectangle
                if x < other_x and x + w > other_x + other_w and y < other_y and y + h > other_y + other_h:
                    level += 1

            # Add the rectangle and its level to the list.
            rectangles.append(((x, y), (x + w, y + h), level))

    return rectangles

def main():
    # Load the image.
    path = "test_rectangles/rect2.png"
    image = cv2.imread(path)

    # Find the nested rectangles.
    rectangles = find_nested_rectangles(image)

    # Print the results.
    for ((x, y), (x_end, y_end), level) in rectangles:
        print(f"Rectangle: ({x}, {y}) - ({x_end}, {y_end}), Level: {level}")

    # Draw the rectangles on the image and label them with their level.
    font = cv2.FONT_HERSHEY_SIMPLEX
    i = 0
    space = 0
    for ((x, y), (x_end, y_end), level) in rectangles:
        # Draw the rectangle
        cv2.rectangle(image, (x, y), (x_end, y_end), (0, 238, 0), 2)
        # Add text showing the level on the top left corner of the rectangle
        if i % 2 != 0:
            space = 0
        else:
            space = 20
        cv2.putText(image, str(level), (x + space, y), font, 0.5, (238, 0, 0), 1, cv2.LINE_AA)
        i += 1
        # print(i)

    # Save the output image
    pattern = r"/([^/.]+)\."
    cv2.imwrite(f"output_rectangles/{re.search(pattern, path).group(1)}_py_output.png", image)

    # Show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

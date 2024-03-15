#include <regex>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat preprocessImage(Mat image) {
    Mat gray, blurred, enhanced;
    
    // Convert the image to grayscale
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    // Apply Gaussian blur to reduce noise
    GaussianBlur(gray, blurred, Size(7, 7), 0);
    
    // Enhance contrast using adaptive histogram equalization
    Ptr<CLAHE> clahe = createCLAHE(10.0, Size(8, 8));
    clahe->apply(blurred, enhanced);

    return enhanced;
}

vector<tuple<Point, Point, int>> findNestedRectangles(Mat image) {
    // Preprocess the image
    Mat preprocessedImage = preprocessImage(image);

    // Threshold the preprocessed image to create a binary image.
    Mat thresh;
    threshold(preprocessedImage, thresh, 127, 255, THRESH_BINARY);

    // Find contours in the binary image.
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Initialize an empty list to store the rectangles and their levels.
    vector<tuple<Point, Point, int>> rectangles;

    // Get the dimensions of the image
    int height = image.rows;
    int width = image.cols;

    // Loop through the contours and find rectangles.
    for (size_t i = 0; i < contours.size(); i++) {
        // Check if the contour is a rectangle.
        vector<Point> approx;
        approxPolyDP(contours[i], approx, 0.01 * arcLength(contours[i], true), true);
        if (approx.size() == 4 && isContourConvex(approx)) {
            // Get the bounding rectangle of the contour.
            Rect rect = boundingRect(contours[i]);
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;

            // Exclude rectangles touching the image edges
            if (x == 0 || y == 0 || x + w == width || y + h == height)
                continue;

            // Calculate the level of nesting for the rectangle.
            int level = 0;
            for (size_t j = 0; j < contours.size(); j++) {
                if (i == j)
                    continue;
                Rect otherRect = boundingRect(contours[j]);
                int other_x = otherRect.x;
                int other_y = otherRect.y;
                int other_w = otherRect.width;
                int other_h = otherRect.height;
                // Check if the current rectangle completely encloses the other rectangle
                if (x < other_x && x + w > other_x + other_w && y < other_y && y + h > other_y + other_h)
                    level++;
            }

            // Add the rectangle and its level to the list.
            rectangles.push_back(make_tuple(Point(x, y), Point(x + w, y + h), level));
        }
    }

    return rectangles;
}

int main() {
    // Load the image.
    string path = "test_rectangles/rect2.png";
    Mat image = imread(path);

    // Find the nested rectangles.
    vector<tuple<Point, Point, int>> rectangles = findNestedRectangles(image);

    // Print the results.
    for (size_t i = 0; i < rectangles.size(); i++) {
        Point pt1 = get<0>(rectangles[i]);
        Point pt2 = get<1>(rectangles[i]);
        int level = get<2>(rectangles[i]);
        cout << "Rectangle: (" << pt1.x << ", " << pt1.y << ") - (" << pt2.x << ", " << pt2.y << "), Level: " << level << endl;
    }

    // Draw the rectangles on the image and label them with their level.
    int i = 0;
    int space = 0;
    for (size_t i = 0; i < rectangles.size(); i++) {
        Point pt1 = get<0>(rectangles[i]);
        Point pt2 = get<1>(rectangles[i]);
        int level = get<2>(rectangles[i]);

        // Draw the rectangle
        rectangle(image, pt1, pt2, Scalar(0, 238, 0), 2);

        // Add text showing the level on the top left corner of the rectangle
        if (i % 2 != 0)
            space = 0;
        else
            space = 20;
        putText(image, to_string(level), Point(pt1.x + space, pt1.y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(238, 0, 0), 1, LINE_AA);
    }

    // Save the output image
    string pattern = "/([^/.]+)\\.";
    smatch match;
    regex_search(path, match, regex(pattern));
    imwrite("output_rectangles/" + match.str(1) + "_cpp_output.png", image);

    // Show the image
    imshow("Image", image);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

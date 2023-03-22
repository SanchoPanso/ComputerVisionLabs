#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>


int main() {
    cv::Mat markerImage;
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::generateImageMarker(dictionary, 23, 200, markerImage, 1);
    cv::imwrite("../cv_lab5/marker23.png", markerImage);
    return 0;
}
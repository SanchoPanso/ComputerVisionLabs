#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include "aruco_samples_utility.hpp"


int main() {

    cv::VideoCapture inputVideo;
    inputVideo.open(1);

    cv::Mat cameraMatrix, distCoeffs;
    float markerLength = 0.05;
    
    // Read camera parameters
    const char* cameraParamsFilename = "../cv_lab5/calibration.xml";
    readCameraParameters(cameraParamsFilename, cameraMatrix, distCoeffs); // This function is implemented in aruco_samples_utility.hpp
    
    // Set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    // Set 3d points of marker cube
    cv::Mat cubePoints3d(8, 1, CV_32FC3);
    
    cubePoints3d.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    cubePoints3d.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength/2.f, markerLength/2.f, 0);
    cubePoints3d.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    cubePoints3d.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    cubePoints3d.ptr<cv::Vec3f>(0)[4] = cv::Vec3f(-markerLength/2.f, markerLength/2.f, markerLength);
    cubePoints3d.ptr<cv::Vec3f>(0)[5] = cv::Vec3f(markerLength/2.f, markerLength/2.f, markerLength);
    cubePoints3d.ptr<cv::Vec3f>(0)[6] = cv::Vec3f(markerLength/2.f, -markerLength/2.f, markerLength);
    cubePoints3d.ptr<cv::Vec3f>(0)[7] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, markerLength);

    
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    // Inintialize video writer
    int frame_width = int(inputVideo.get(3));
    int frame_height = int(inputVideo.get(4));
    cv::Size frame_size(frame_width, frame_height);
    cv::VideoWriter writer("C:\\Users\\HP\\Desktop\\cv_lab5.mp4", 
                           cv::VideoWriter::fourcc('P','I','M','1'), 
                           20, frame_size);
    
    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        
        detector.detectMarkers(image, corners, ids);
        
        // If at least one marker detected
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
            int nMarkers = corners.size();
            std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);
        
            // Calculate pose for each marker
            for (int i = 0; i < nMarkers; i++) {
                solvePnP(objPoints, corners.at(i), cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
            }
        
            // Draw cube for each marker
            for(unsigned int i = 0; i < ids.size(); i++) {
                std::vector<cv::Point2f> cubePoints2d;
                cv::projectPoints(cubePoints3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs, cubePoints2d);

                // Draw horizontal lines of the cube          
                for (int j = 0; j < 4; j++) {
                    cv::line(imageCopy, 
                             cubePoints2d[j % 4], 
                             cubePoints2d[(j + 1) %4], 
                             cv::Scalar(0, 255, 0), 2);
                    cv::line(imageCopy, 
                             cubePoints2d[j % 4 + 4], 
                             cubePoints2d[(j + 1) %4 + 4], 
                             cv::Scalar(0, 255, 0), 2);
                }
                
                // Draw vertical lines of the cube          
                for (int j = 0; j < 4; j++) {
                    cv::line(imageCopy, cubePoints2d[j], cubePoints2d[j + 4], cv::Scalar(0, 255, 0), 2);
                }
            }
        }
        // Show resulting image and close window
        cv::imshow("out", imageCopy);
        
        // Write image to video
        writer.write(imageCopy);
        
        // If Esc - break
        char key = (char) cv::waitKey(1);
        if (key == 27)
            break;
    }

    inputVideo.release();
    return 0;
}




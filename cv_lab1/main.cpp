#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <cmath>

#define MAIN_BACKGROUND_PATH "..\\cv_lab1\\main_background.jpg"
#define RESULT_PATH "..\\cv_lab1\\result.jpg"

// #define M_PI 3.14

void add_weighted(cv::Mat &src1, cv::Mat &src2, double alpha, cv::Mat &dst) {

    int num_rows = src1.rows;
    int num_cols = src1.cols *  src1.channels();

    for(int i = 0; i < num_rows; ++i) {
        auto src1_p = src1.ptr<uchar>(i);
        auto src2_p = src2.ptr<uchar>(i);
        auto dst_p = dst.ptr<uchar>(i);
        for (int j = 0; j < num_cols; ++j) {
            dst_p[j] = static_cast<int>(src1_p[j] * alpha + src2_p[j] * (1 - alpha));
        }
    }
}


int main() {

    double freq = 0.05;
    double amplitude = 20;

    cv::Scalar path_color(0, 0, 255);
    cv::Scalar obj_color(0,255,0);
    cv::Scalar bg_color(255, 0, 0);

    cv::Mat main_background = cv::imread(MAIN_BACKGROUND_PATH);
    int height = main_background.rows;
    int width = main_background.cols;
    cv::Mat additional_background(height, width, CV_8UC3, bg_color);

    for (int max_col = 0; max_col < width; max_col++) {

        // Create black img
        cv::Mat img(height, width, CV_8UC3);
        add_weighted(main_background, additional_background, 0.5 * (1 + sin(freq * max_col)), img);

        // Draw gone path
        for (int col = 0; col < max_col; col++){
            int row = height / 2 + (int) (amplitude * sin(freq * col));
            cv::circle(img, cv::Point(col, row), 1, path_color, -1);
        }

        // Create rotated rectangle
        int max_row = height / 2 + (int) (amplitude * sin(freq * max_col));
        cv::RotatedRect rot_rect(cv::Point(max_col, max_row),
                                 cv::Size(20, 20),
                                 amplitude * freq * cos(freq * max_col) * 180 / 3.14);

        // Draw rotated rectangle
        cv::Point2f vertices[4];
        rot_rect.points(vertices);
        for (int i = 0; i < 4; i++)
            line(img, vertices[i], vertices[ (i + 1) % 4], obj_color, 2);

        imshow("Display window", img);

        // Press Esc - end
        if (cv::waitKey(5) == 27){
            cv::imwrite(RESULT_PATH, img);
            break;
        }

        // if object is on center - end
        if (max_col > width / 2){
            cv::imwrite(RESULT_PATH, img);
            break;
        }
    }

    return 0;
}

#include <iostream>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include "cvDirectory.h"

#define IMG_ZADAN_PATH      "..\\cv_lab3\\img_zadan\\"

#define TASK_1_IMAGE_DIR    IMG_ZADAN_PATH"allababah\\"
#define TASK_2_IMAGE_DIR    IMG_ZADAN_PATH"teplovizor\\"
#define TASK_3_IMAGE_DIR    IMG_ZADAN_PATH"roboti\\"
#define TASK_4_IMAGE_DIR    IMG_ZADAN_PATH"gk\\"


struct Area {
    std::vector<cv::Point> contour;
    cv::Point center;
    Area(std::vector<cv::Point> contour, cv::Point center);
};

void task1();
void task2();
void task3();
void task4();

cv::Point find_lamp_center(cv::Mat &img, 
                           cv::Scalar lamp_lower = cv::Scalar(0, 0, 200), 
                           cv::Scalar lamp_upper = cv::Scalar(180, 10, 255));
std::vector<Area> get_area_centers(cv::Mat &img,
                                   cv::Mat &mask,
                                   cv::Scalar lower = cv::Scalar(0, 0, 100),
                                   cv::Scalar upper = cv::Scalar(50, 255, 255), 
                                   double contour_area_thresh = 50.0);
void draw_ares_centers(cv::Mat &img, std::vector<Area> areas);


int main(int, char**) {
    
    task1();
    task2();
    task3();
    task4();

    return 0;
}


Area::Area(std::vector<cv::Point> contour, cv::Point center) {
        this->contour = contour;
        this->center = center;
}


void task1() {
    std::string image_dir = std::string(TASK_1_IMAGE_DIR);
    std::vector<std::string> fnms = cv::Directory::GetListFiles(image_dir, std::string("*.jpg"), false);

    for (auto & fnm : fnms) {
        cv::Mat img = cv::imread(image_dir + fnm);

        cv::Scalar lower(0, 0, 200);
        cv::Scalar upper(180, 255, 255);

        cv::Mat mask(img.rows, img.cols, CV_8U);
        auto areas = get_area_centers(img, mask, lower, upper);
        draw_ares_centers(img, areas);
        cv::imshow("img", img);
        cv::waitKey();

        cv::imwrite("task_1.jpg", img);
    }

    cv::destroyAllWindows();
}


void task2() {
    std::string image_dir = std::string(TASK_2_IMAGE_DIR);
    std::vector<std::string> fnms = cv::Directory::GetListFiles(image_dir, std::string("*.jpg"), false);

    for (auto & fnm : fnms) {
        cv::Mat img = cv::imread(image_dir + fnm);
        cv::Mat mask(img.rows, img.cols, CV_8U);
        auto areas = get_area_centers(img, mask);
        draw_ares_centers(img, areas);
        cv::imshow("img", img);
        cv::waitKey();

        cv::imwrite("task_2.jpg", img);
    }

    cv::destroyAllWindows();
}


void task3() {
    std::string image_dir = std::string(TASK_3_IMAGE_DIR);
    std::vector<std::string> fnms = cv::Directory::GetListFiles(image_dir, std::string("*.jpg"), false);
    std::vector<std::vector<cv::Scalar>> team_colors_ranges = {
        {{0, 0, 10}, {10, 255, 255}},
        {{60, 50, 140}, {85, 255, 255}},
        {{80, 0, 10}, {110, 255, 255}},
    }; 
    std::vector<cv::Scalar> team_colors = {
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
    };

    for (auto & fnm : fnms) {
        cv::Mat img = cv::imread(image_dir + fnm);
        cv::Mat vis_img = img.clone();

        cv::Point lamp_center = find_lamp_center(img);
        cv::rectangle(img, cv::Rect(lamp_center.x - 40, lamp_center.y - 55, 70, 60), cv::Scalar(0, 0, 0), -1);

        cv::circle(vis_img, lamp_center, 5, cv::Scalar(0, 64, 64), -1);

        for (int i = 0; i < team_colors_ranges.size(); i++) {
            auto lower = team_colors_ranges[i][0];
            auto upper = team_colors_ranges[i][1];

            cv::Mat mask(img.rows, img.cols, CV_8U);
            auto areas = get_area_centers(img, mask, lower, upper, 100.0);

            int nearest_idx = -1;
            int min_dist = -1;
            
            for (int j = 0; j < areas.size(); j++) {
                Area cur_area = areas[j];
                
                double lamp_x = lamp_center.x;
                double lamp_y = lamp_center.y;

                double robot_x = cur_area.center.x;
                double robot_y = cur_area.center.y; 

                double cur_dist = sqrt(pow((robot_x - lamp_x), 2) + pow((robot_y - lamp_y), 2));  
                if (j == 0 || cur_dist < min_dist) {
                    min_dist = cur_dist;
                    nearest_idx = j;
                }
            }
            cv::circle(vis_img, areas[nearest_idx].center, 5, team_colors[i], -1);
            cv::circle(vis_img, areas[nearest_idx].center, 5, cv::Scalar(0, 0, 0), 1);        
        }

        cv::imshow("vis_img", vis_img);
        cv::waitKey();

        cv::imwrite("task_3.jpg", vis_img);
    }
    cv::destroyAllWindows();
}

void task4() {
    std::string image_path = std::string(TASK_4_IMAGE_DIR) + std::string("gk.jpg");
    std::string template_path = std::string(TASK_4_IMAGE_DIR) + std::string("gk_tmplt.jpg");

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat tmplt = cv::imread(template_path, cv::IMREAD_GRAYSCALE);

    cv::Mat img_hsv(img.rows, img.cols, CV_8UC3);
    cv::Mat mask(img.rows, img.cols, CV_8U);
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
    cv::inRange(img_hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 250, 250), mask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::imshow("mask", mask);
    cv::imshow("tmplt", tmplt);
    cv::waitKey();

    std::vector<std::vector<cv::Point>> tmplt_contours;
    cv::findContours(tmplt, tmplt_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    
    std::vector<cv::Point> tmplt_contour;
    double max_area = 0;
    for (auto & cnt : tmplt_contours) {
        double cur_area = cv::contourArea(cnt);
        if (cur_area > max_area) {
            max_area = cur_area;
            tmplt_contour = cnt;
        }
    }

    for (auto & cnt : contours) {
        if (cv::contourArea(cnt) < 10) {
            continue;
        }
        double metric = cv::matchShapes(cnt, tmplt_contour, cv::CONTOURS_MATCH_I2, 0);
        
        std::vector<std::vector<cv::Point>> contours = {cnt};
        if (metric < 0.2) {
            cv::drawContours(img, contours, 0, cv::Scalar(0, 255, 0), 4);
        } else {
            cv::drawContours(img, contours, 0, cv::Scalar(0, 0, 255), 4);
        }
    }

    cv::imshow("img", img);
    cv::waitKey();

    cv::imwrite("task_4.jpg", img);

    cv::destroyAllWindows();
}


cv::Point find_lamp_center(cv::Mat &img, cv::Scalar lamp_lower, cv::Scalar lamp_upper) {

    cv::Mat mask(img.rows, img.cols, CV_8U);
    auto lamp_areas = get_area_centers(img, mask, lamp_lower, lamp_upper);
    
    if (lamp_areas.size() == 0) {
        return cv::Point(-1, -1);
    }

    cv::Point lamp_center = lamp_areas[0].center;
    return lamp_center;
}


void draw_ares_centers(cv::Mat &img, std::vector<Area> areas) {
    
    for (int i = 0; i < areas.size(); i++) {
        std::vector<std::vector<cv::Point>> contours = {areas[i].contour};
        cv::Point center = areas[i].center;

        cv::drawContours(img, contours, 0, cv::Scalar(0, 255, 0), 2);
        cv::circle(img, center, 5, cv::Scalar(0, 0, 255), -1);
    }
    
}

std::vector<Area> get_area_centers(cv::Mat &img,
                                   cv::Mat &mask,
                                   cv::Scalar lower, 
                                   cv::Scalar upper, 
                                   double contour_area_thresh) {

    cv::Mat img_hsv(img.rows, img.cols, CV_8UC3);

    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);
    cv::inRange(img_hsv, lower, upper, mask);

    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    std::vector<std::vector<cv::Point>> contours; 
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
 
    std::vector<Area> areas;
    for (int i = 0; i < contours.size(); i++) {
        
        auto cnt = contours[i];
        if (cv::contourArea(cnt) < contour_area_thresh) {
            continue;
        }

        cv::Moments moments = cv::moments(cnt);
        int xc = moments.m10 / moments.m00;
        int yc = moments.m01 / moments.m00;
        
        Area area(cnt, cv::Point(xc, yc));
        areas.push_back(area);
    }

    return areas;
}


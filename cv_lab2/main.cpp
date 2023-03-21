#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <cmath>

#include "TickMeter.h"

#define SOURCE_IMAGE_PATH "../cv_lab2/lenna.png"


cv::Mat blur(cv::Mat &src, cv::Size ksize);
cv::Mat gauss_unsharp_mask(cv::Mat &src, cv::Size gauss_ksize);
cv::Mat box_unsharp_mask(cv::Mat &src, cv::Size ksize);
cv::Mat laplace(cv::Mat &src);
cv::Mat laplace_unsharp_mask(cv::Mat &src, double sharp_coef = 1.0);
cv::Mat logarithmic_transform(cv::Mat &src, double c = 1.0);

double getSimilarityPercentage(cv::Mat &img1, cv::Mat &img2);
cv::Mat get_diff_image(cv::Mat &img1, cv::Mat &img2);

void filter2D_int32(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel);
void convolve(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel, int x, int y);
int find_reflected101_index(int orig_idx, int boundary_idx);
cv::Mat get_roi_with_border_reflect101(cv::Mat &src, int x, int y, cv::Size ksize);


int main() {

    cv::Mat img = cv::imread(SOURCE_IMAGE_PATH);
    cv::resize(img, img, cv::Size(400, 400));

    // // Task 1    
    // // Blur image with custom function by 3x3 kernel
    // auto custom_blur_img = blur(img, cv::Size(3, 3));
    
    // // Show original and gotten images
    // cv::imshow("img", img);
    // cv::imshow("custom_blur_img", custom_blur_img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();


    // // Task2
    // cv::Mat opencv_blur_img(img.rows, img.cols, img.type());
    // cv::blur(img, opencv_blur_img, cv::Size(3, 3));
    // auto blur_diff_img = get_diff_image(opencv_blur_img, custom_blur_img);

    // std::cout << "Difference: " << get_diff_percentage(custom_blur_img, opencv_blur_img) << "%" << std::endl;

    // cv::imshow("opencv_blur_img", opencv_blur_img);
    // cv::imshow("custom_blur_img", custom_blur_img);
    // cv::imshow("blur_diff_img", logarithmic_transform(blur_diff_img, 20));
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // // Task 3
    // TickMeter tick_meter;
    
    // tick_meter.start();
    // auto custom_blur_img_tmp = blur(img, cv::Size(3, 3));
    // tick_meter.stop();
    // std::cout << "Custom blur ticks: " << tick_meter.getTimeTicks() << '\n';
    // tick_meter.reset();

    // tick_meter.start();
    // cv::blur(img, opencv_blur_img, cv::Size(3, 3));
    // tick_meter.stop();
    // std::cout << "OpenCV blur ticks: " << tick_meter.getTimeTicks() << '\n';
    // tick_meter.reset();

    // // Task 4
    // cv::Mat opencv_gaussian_blur_img(img.rows, img.cols, img.type());
    // cv::GaussianBlur(img, opencv_gaussian_blur_img, cv::Size(3, 3), 1);
    // cv::Mat gaussian_box_diff_img = get_diff_image(opencv_gaussian_blur_img, opencv_blur_img); 
    // gaussian_box_diff_img = logarithmic_transform(gaussian_box_diff_img, 10);

    // cv::imshow("opencv_blur_img", opencv_blur_img);
    // cv::imshow("opencv_gaussian_blur_img", opencv_gaussian_blur_img);
    // cv::imshow("gaussian_box_diff_img", gaussian_box_diff_img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // // Task 5
    // auto gauss_unsharp_img = gauss_unsharp_mask(img, cv::Size(3, 3));
    // auto box_unsharp_img = box_unsharp_mask(img, cv::Size(3, 3));
    // auto diff_img = get_diff_image(gauss_unsharp_img, box_unsharp_img);
    // diff_img = logarithmic_transform(diff_img, 10);

    // cv::imshow("gauss_unsharp_img", gauss_unsharp_img);
    // cv::imshow("box_unsharp_img", box_unsharp_img);
    // cv::imshow("diff_img", diff_img);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // Task 6
    auto laplace_img = laplace(img);
    cv::imshow("laplace_img", laplace_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::imwrite("laplace_img.jpg", laplace_img);

    // Task 7
    cv::imshow("img", img);
    auto laplace_unsharp_img = laplace_unsharp_mask(img, 2.0);
    
    // auto gauss_laplace_diff_img = get_diff_image(gauss_unsharp_img, laplace_unsharp_img);
    // gauss_laplace_diff_img = logarithmic_transform(gauss_laplace_diff_img, 10);
    
    // auto box_laplace_diff_img = get_diff_image(box_unsharp_img, laplace_unsharp_img);
    // box_laplace_diff_img = logarithmic_transform(box_laplace_diff_img, 10);
    
    cv::imshow("laplace_unsharp_img", laplace_unsharp_img);
    // cv::imshow("box_laplace_diff_img", box_laplace_diff_img);
    // cv::imshow("gauss_laplace_diff_img", gauss_laplace_diff_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::imwrite("laplace_unsharp_img.jpg", laplace_unsharp_img);

    return 0;
}


cv::Mat blur(cv::Mat &src, cv::Size ksize) {

    int num_rows = src.rows;
    int num_cols = src.cols;

    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    cv::Mat kernel = cv::Mat::ones(ksize.width, ksize.height, CV_32S);
    filter2D_int32(src, dst, kernel);

    return dst;
}


void convolve(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel, int x, int y) {
    int num_of_channels = src.channels();

    int kernel_height = kernel.rows;
    int kernel_width = kernel.cols;

    auto roi = get_roi_with_border_reflect101(src, x, y, cv::Size(kernel_width, kernel_height));

    std::vector<int> sums(num_of_channels, 0);
    int divider = 0;

    for (int roi_y = 0; roi_y < roi.rows; roi_y++) {
        auto p = roi.ptr<uchar>(roi_y);

        for (int roi_x = 0; roi_x < roi.cols; roi_x++) {
            divider += kernel.at<int>(roi_y, roi_x);
            for (int channel = 0; channel < num_of_channels; channel++) {
                sums[channel] += p[num_of_channels * roi_x + channel] * kernel.at<int>(roi_y, roi_x);
            }
        }
    }


    auto dst_p = dst.ptr<uchar>(y);
    divider = (divider == 0) ? 1 : divider;
    for (int channel = 0; channel < num_of_channels; channel++) {
        int round_summand = ((sums[channel] % divider > divider / 2) ? 1 : 0); /// fix round
        dst_p[3 * x + channel] = std::max(0, sums[channel] / divider + round_summand);
    }
}


void filter2D_int32(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel) {
    int num_rows = src.rows;
    int num_cols = src.cols;

    for(int y = 0; y < num_rows; ++y) {
        for (int x = 0; x < num_cols; ++x) {
            convolve(src, dst, kernel, x, y);
        }
    }
}


cv::Mat logarithmic_transform(cv::Mat &src, double c) {
    
    uint8_t table_data[256];
    for (int i = 0; i < 256; i++) {
        table_data[i] = (uint8_t) (c * log2(1.0 + i));
    }

    cv::Mat table(1, 256, CV_8U, table_data);
    cv::Mat dst(src.rows, src.cols, src.type());
    cv::LUT(src, table, dst);

    return dst;
}


double getSimilarityPercentage(cv::Mat &img1, cv::Mat &img2) {

    int num_of_channels = img1.channels();
    int sum_of_matches = 0;

    for (int y = 0; y < img1.rows; y++) {
        auto p1 = img1.ptr<uchar>(y);
        auto p2 = img2.ptr<uchar>(y);

        for (int x = 0; x < img1.cols; x++) {
            for (int channel = 0; channel < num_of_channels; channel++) {
                auto value1 = p1[x * num_of_channels + channel];
                auto value2 = p2[x * num_of_channels + channel];
                if (value1 == value2) {
                    sum_of_matches++;
                }
            }
        }
    }

    return double(sum_of_matches) / (img1.cols * img1.rows * num_of_channels) * 100.0;
}

cv::Mat get_diff_image(cv::Mat &img1, cv::Mat &img2) {
    int num_of_channels = img1.channels();
    int sum_of_matches = 0;
    cv::Mat result(img1.rows, img1.cols, img1.type());

    for (int y = 0; y < img1.rows; y++) {
        auto p_img1 = img1.ptr<uchar>(y);
        auto p_img2 = img2.ptr<uchar>(y);
        auto p_result = result.ptr<uchar>(y);

        for (int x = 0; x < img1.cols; x++) {
            for (int channel = 0; channel < num_of_channels; channel++) {
                auto value_img1 = p_img1[x * num_of_channels + channel];
                auto value_img2 = p_img2[x * num_of_channels + channel];
                p_result[x * num_of_channels + channel] = abs(value_img2 - value_img1);
            }
        }
    }

    return result;
}


cv::Mat gauss_unsharp_mask(cv::Mat &src, cv::Size ksize) {

    cv::Mat blurred(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    cv::GaussianBlur(src, blurred, ksize, ksize.width / 3.0);
    cv::Mat dst = src + (src - blurred);
    return dst;
}

cv::Mat box_unsharp_mask(cv::Mat &src, cv::Size ksize) {

    cv::Mat blurred(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    cv::blur(src, blurred, ksize);
    cv::Mat dst = src + (src - blurred);
    return dst;
}


cv::Mat laplace(cv::Mat &src) {

    cv::Mat laplaced(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    int32_t kernel_data[9] = {0, 1, 0,
                              1, -4, 1,
                              0, 1, 0};
    cv::Mat kernel(3, 3, CV_32S, kernel_data);
    filter2D_int32(src, laplaced, kernel);
    //cv::medianBlur(laplaced, laplaced, 3);
    cv::threshold(laplaced, laplaced, 20, 255, cv::THRESH_TOZERO);
    cv::blur(laplaced, laplaced, cv::Size(3, 3));
    
    //cv::threshold(laplaced, laplaced, 150, 255, cv::THRESH_TOZERO);
    
    return laplaced;
}


cv::Mat laplace_unsharp_mask(cv::Mat &src, double sharp_coef) {

    //cv::GaussianBlur(src, src, cv::Size(3, 3), 1);
    cv::Mat laplaced = laplace(src);

    cv::Mat double_laplaced;
    laplaced.convertTo(double_laplaced, CV_64FC3);
    cv::Mat weighted_double_laplaced = sharp_coef * double_laplaced;
    weighted_double_laplaced.convertTo(laplaced, CV_8UC3);

    cv::Mat dst = src - laplaced; 

    // auto k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // cv::morphologyEx(dst,dst, cv::MORPH_CLOSE, k);
    

    return dst;
}


int find_reflected101_index(int orig_idx, int boundary_idx) {
    int doubled_res_idx = orig_idx % (boundary_idx * 2);

    if (doubled_res_idx < 0) {
        doubled_res_idx = abs(doubled_res_idx);
    }

    if (doubled_res_idx < boundary_idx) {
        return doubled_res_idx;
    }
    return 2 * boundary_idx - doubled_res_idx - 1;
}


cv::Mat get_roi_with_border_reflect101(cv::Mat &src, int x, int y, cv::Size ksize) {

    // temporary fix of even numbers
    if (ksize.width % 2 == 0) {
        ksize.width++;
    }
    if (ksize.height % 2 == 0) {
        ksize.height++;
    }

    int min_roi_x = x - ksize.width / 2;
    int max_roi_x = x + ksize.width / 2;
    int min_roi_y = y - ksize.height / 2;
    int max_roi_y = y + ksize.height / 2;

    if ((min_roi_x >= 0) && (max_roi_x < src.cols) && (min_roi_y >= 0) && (max_roi_y < src.rows)) {
        return src(cv::Rect(min_roi_x, min_roi_y, ksize.width, ksize.height));
    }

    cv::Mat roi(ksize.width, ksize.height, CV_8UC3);
    for(int local_roi_y = 0; local_roi_y < ksize.height; local_roi_y++) {

        int src_y = find_reflected101_index(min_roi_y + local_roi_y, src.rows);
        auto roi_p = roi.ptr<uchar>(local_roi_y);
        auto src_p = src.ptr<uchar>(src_y);

        for (int local_roi_x = 0; local_roi_x < ksize.width; local_roi_x++) {
            int src_x = find_reflected101_index(min_roi_x + local_roi_x, src.cols);

            for (int c = 0; c < src.channels(); c++) {
                roi_p[local_roi_x * src.channels() + c] = src_p[src_x * src.channels() + c];
            }
        }
    }

    return roi;
}



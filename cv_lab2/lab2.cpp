#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <thread>

#include <iostream>
#include <cmath>


void filter2D_uint8(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel);
int find_reflected101_index(int orig_idx, int boundary_idx);
cv::Mat get_roi_with_border_reflect101(cv::Mat &src, int x, int y, cv::Size ksize);
cv::Mat unsharp_mask(cv::Mat &src, cv::Size &gauss_ksize);


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
        return src(cv::Rect(min_roi_x, min_roi_y, max_roi_x - min_roi_x, max_roi_y - min_roi_y));
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


void box_filter(cv::Mat &img, cv::Mat &roi, int x, int y) {

    int num_of_channels = roi.channels();
    std::vector<int> sums(num_of_channels, 0);

    for (int roi_y = 0; roi_y < roi.rows; roi_y++) {
        auto p = roi.ptr<uchar>(roi_y);

        for (int roi_x = 0; roi_x < roi.cols; roi_x++) {
            for (int channel = 0; channel < num_of_channels; channel++) {
                sums[channel] += p[num_of_channels * roi_x + channel];
            }
        }
    }


    auto p = img.ptr<uchar>(y);
    for (int channel = 0; channel < num_of_channels; channel++) {
        p[3 * x + channel] = sums[channel] / (roi.rows * roi.cols);
    }
}


cv::Mat custom_blur(cv::Mat &src, cv::Size ksize) {

    int num_rows = src.rows;
    int num_cols = src.cols;

    cv::Mat dst(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    cv::Mat kernel = cv::Mat::ones(ksize.width, ksize.height, CV_32S);
    filter2D_uint8(src, dst, kernel);

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
                sums[channel] += p[num_of_channels * roi_x + channel];
            }
        }
    }


    auto dst_p = dst.ptr<uchar>(y);
    divider = (divider == 0) ? 1 : divider;
    for (int channel = 0; channel < num_of_channels; channel++) {
        dst_p[3 * x + channel] = sums[channel] / divider;
    }
}


void filter2D_uint8(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel) {
    int num_rows = src.rows;
    int num_cols = src.cols;

    for(int y = 0; y < num_rows; ++y) {
        for (int x = 0; x < num_cols; ++x) {
            convolve(src, dst, kernel, x, y);
        }
    }
}

double get_diff_percentage(cv::Mat &img1, cv::Mat &img2) {

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


cv::Mat unsharp_mask(cv::Mat &src, cv::Size gauss_ksize) {

    cv::Mat blurred(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    cv::GaussianBlur(src, blurred, gauss_ksize, gauss_ksize.width / 3.0);
    cv::Mat dst = src + (src - blurred);
    return dst;
}


cv::Mat laplace(cv::Mat &src, double sharp_coef = 0.5) {

    cv::Mat laplaced(src.rows, src.cols, src.type(), cv::Scalar(0, 0, 0));
    int32_t kernel_data[9] = {0, 1, 0,
                              1, -4, 1,
                              0, 1, 0};
    cv::Mat kernel(3, 3, CV_32S, kernel_data);
    filter2D_uint8(src, laplaced, kernel);

    cv::Mat double_laplaced;
    laplaced.convertTo(double_laplaced, CV_64FC3);
    cv::Mat weighted_double_laplaced = sharp_coef * double_laplaced;
    weighted_double_laplaced.convertTo(laplaced, CV_8UC3);

    cv::Mat dst = src - laplaced;
    return dst;
}


int main() {

    // cv::Mat img = cv::imread("../lenna.png");
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
    cv::rectangle(img, cv::Point(2, 2), cv::Point(10, 10), cv::Scalar(0, 0, 255), -1);
    cv::rectangle(img, cv::Point(90, 90), cv::Point(98, 98), cv::Scalar(0, 0, 255), -1);

    auto roi1 = get_roi_with_border_reflect101(img, 0, 0, cv::Size(51, 51));
    auto roi2 = get_roi_with_border_reflect101(img, 99, 99, cv::Size(51, 51));
    auto blurred = custom_blur(img, cv::Size(5, 5));
    auto unsharp_masked = unsharp_mask(blurred, cv::Size(5, 5));
    auto blur_diff = get_diff_image(img, blurred);
    auto laplaced = laplace(blurred);

    std::cout << get_diff_percentage(unsharp_masked, blurred) << std::endl;

    imshow("img", img);
    imshow("roi1", roi1);
    imshow("roi2", roi2);
    imshow("blurred", blurred);
    imshow("unsharp_masked", unsharp_masked);
    imshow("blur_diff", blur_diff);
    imshow("laplaced", laplaced);

    cv::waitKey(0);

    return 0;
}

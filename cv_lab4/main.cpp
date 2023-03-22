#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <complex>

#include <opencv2/opencv.hpp>


enum DftFlags {
    /** performs an inverse 1D or 2D transform instead of the default forward
        transform. */
    DFT_INVERSE        = 1 << 0,
    
    DFT_COMPLEX_OUTPUT = 0 << 1,
    DFT_REAL_OUTPUT    = 1 << 1,
};


void task_1();  // dft
void task_2();  // fft
void task_3();  // speed
void task_4();  // conv
void task_5();  // cut
void task_6();  // corr

void mulSpectrumShow(cv::InputArray orig_img, cv::InputArray filter);


void test_dft();
void test_idft();
void test_fft();
void test_ifft();


/**
 * @brief Calculates the Discrete Fourier Transform of a 1D array using direct method.
 * 
 * @param src floating-point (CV_64F) real or complex array. It must be one row or one colunm, exception raises otherwise
 * @param dst output array whose size and type depend on the flags.
 * @param flag operation flags (see #DftFlags).
 * @param w input array of precomputed coefficients "W" for dft (you SHOULD NOT (!!!) set it manually, 
 * it is used for dft2D for optimization)
 */
void dft1D(cv::InputArray src, cv::OutputArray dst, int flag = 0, cv::InputArray w = cv::noArray());

/**
 * @brief Calculates the Discrete Fourier Transform of a 2D array using direct method.
 * 
 * @param src floating-point (CV_64F) real or complex array.
 * @param dst output array whose size and type depend on the flags.
 * @param flag operation flags (see #DftFlags).
 */
void dft2D(cv::InputArray src, cv::OutputArray dst, int flag = 0);

/**
 * @brief Calculates the Fast Fourier Transform of a 1D array.
 * 
 * @param src floating-point (CV_64F) real or complex array. It must be one row or one colunm, exception raises otherwise.
 * @param dst output array whose size and type depend on the flags.
 * @param flag operation flags (see #DftFlags).
 */
void fft1D(cv::InputArray src, cv::OutputArray dst, int flag = 0);


/**
 * @brief Calculates the Fast Fourier Transform of a 2D array.
 * 
 * @param src floating-point (CV_64F) real or complex array.
 * @param dst output array whose size and type depend on the flags.
 * @param flag operation flags (see #DftFlags).
 */
void fft2D(cv::InputArray src, cv::OutputArray dst, int flag = 0);

/**
 * @brief Expand both sources to one size that is suitable for spectrum multiplication. Empty space filled with zeros.
 * 
 * @param src1 the first input array
 * @param src2 the second input array
 * @param dst1 expanded src1 to suitable size
 * @param dst2 expanded src2 to suitable size
 */
void PrepareForMullSpectrum(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst1, cv::OutputArray dst2);
void convolveDFT(cv::InputArray A, cv::InputArray B, cv::OutputArray C);

void calc_dft_coef(int n, int sign, cv::OutputArray w);

/**
 * @brief Rearrange the quadrants of Fourier image so that the origin is at the image center.	
 * 
 * @param magI input Mat of magnitude of fourier image. It is transformed inplace.
 */
void krasivSpektr(cv::Mat &magI);

/**
 * @brief Get the pretty fourier image
 * 
 * @param fourier 2-channel floatting point input array of fourier image 
 * @return cv::Mat 1-channel floatting-point normalized magnitude with swapped quadrants (origin in the center) 
 */
cv::Mat getPrettyFourier(cv::Mat fourier);

template <typename T> void printMat(cv::Mat &img);

int get_optimal_size(int size);
void createOptimalSizedMat(cv::InputArray src, cv::OutputArray dst);

int rev(int num, int lg_n);
unsigned char reverse(unsigned char b);
double getSimilarityPercentage(cv::Mat &img1, cv::Mat &img2);


int main() {
    task_5();
    return 0;
}


void task_1() {

    // Read image in grayscale and convert to double
    std::string image_path = "D:\\CodeProjects\\C_CPP_Projects\\ComputerVisionLabs\\cv_lab4\\250px-Fourier2.jpg";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_64F);
    
    // Make fourier image
    cv::Mat custom_fourier, opencv_fourier;
    dft2D(img, custom_fourier, DFT_COMPLEX_OUTPUT);         // Custom dft
    cv::dft(img, opencv_fourier, cv::DFT_COMPLEX_OUTPUT);   // OpenCV dft

    // Prettify fourier image
    cv::Mat custom_pretty_fourier = getPrettyFourier(custom_fourier);
    cv::Mat opencv_pretty_fourier = getPrettyFourier(opencv_fourier);
    img.convertTo(img, CV_8U);

    // Find difference between dft methods
    double dft_diff = getSimilarityPercentage(custom_pretty_fourier, opencv_pretty_fourier);
    std::cout << "DFT difference: " << dft_diff << "%" << std::endl;

    // Apply custom inverse dft to get original image 
    cv::Mat img_from_custom_furier;
    dft2D(custom_fourier, img_from_custom_furier, DFT_REAL_OUTPUT | DFT_INVERSE);
    
    img_from_custom_furier.convertTo(img_from_custom_furier, CV_8U);
    img.convertTo(img, CV_8U);

    // Find difference between original image and image transformed twice
    double idft_diff = getSimilarityPercentage(img, img_from_custom_furier);
    std::cout << "DFT difference: " << idft_diff << "%" << std::endl;
    
    cv::imshow("img", img);
    cv::imshow("custom_fourier", custom_pretty_fourier);
    cv::imshow("opencv_fourier", opencv_pretty_fourier);
    cv::imshow("img_from_custom_furier", img_from_custom_furier);

    cv::waitKey();
}


void task_2() {
    // Read image in grayscale and convert to double
    std::string image_path = "D:\\CodeProjects\\C_CPP_Projects\\ComputerVisionLabs\\cv_lab4\\250px-Fourier2.jpg";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_64F);

    // expand image to degree of two for comfortable comparison
    createOptimalSizedMat(img, img);
    
    // Make fourier image
    cv::Mat custom_fourier, opencv_fourier;
    fft2D(img, custom_fourier, DFT_COMPLEX_OUTPUT);         // Custom dft
    cv::dft(img, opencv_fourier, cv::DFT_COMPLEX_OUTPUT);   // OpenCV dft

    // Prettify fourier image
    cv::Mat custom_pretty_fourier = getPrettyFourier(custom_fourier);
    cv::Mat opencv_pretty_fourier = getPrettyFourier(opencv_fourier);
    img.convertTo(img, CV_8U);

    // Find difference between dft methods
    double dft_diff = getSimilarityPercentage(custom_pretty_fourier, opencv_pretty_fourier);
    std::cout << "DFT difference: " << dft_diff << "%" << std::endl;

    // Apply custom inverse dft to get original image 
    cv::Mat img_from_custom_furier;
    fft2D(custom_fourier, img_from_custom_furier, DFT_REAL_OUTPUT | DFT_INVERSE);
    
    img_from_custom_furier.convertTo(img_from_custom_furier, CV_8U);
    img.convertTo(img, CV_8U);

    // Find difference between original image and image transformed twice
    double idft_diff = getSimilarityPercentage(img, img_from_custom_furier);
    std::cout << "DFT difference: " << idft_diff << "%" << std::endl;
    
    cv::imshow("img", img);
    cv::imshow("custom_fourier", custom_pretty_fourier);
    cv::imshow("opencv_fourier", opencv_pretty_fourier);
    cv::imshow("img_from_custom_furier", img_from_custom_furier);

    cv::waitKey();
}

void task_4() {
    // Read image in grayscale and convert to double
    std::string image_path = "D:\\CodeProjects\\C_CPP_Projects\\ComputerVisionLabs\\cv_lab4\\fftdemo.jpg";
    cv::Mat orig_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    orig_img.convertTo(orig_img, CV_64F);

    // Vertical Sobel
    double vsobel_data[] = {-1, 0, 1, 
                            -2, 0, 2,
                            -1, 0, 1,};
    cv::Mat vsobel(3, 3, CV_64F, vsobel_data);
    mulSpectrumShow(orig_img, vsobel);

    // Horizontal Sobel
    double hsobel_data[] = {-1, -2, -1, 
                            0, 0, 0,
                            1, 2, 1,};
    cv::Mat hsobel(3, 3, CV_64F, hsobel_data);
    mulSpectrumShow(orig_img, hsobel);

    // Box Filter
    cv::Mat box_filter(3, 3, CV_64F, 1.0 / 9.0);
    mulSpectrumShow(orig_img, box_filter);

    // Laplace
    double laplace_data[] = {0, -1, 0, 
                            -1, 4, -1,
                            0, -1, 0,};
    cv::Mat laplace(3, 3, CV_64F, laplace_data);
    mulSpectrumShow(orig_img, laplace);

}


void task_5() {
    std::string image_path = "/home/student2/Pictures/Screenshot from 2023-03-22 17-22-43.png";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_64F);

    cv::Mat img_fourier;
    cv::dft(img, img_fourier, cv::DFT_COMPLEX_OUTPUT);

    cv::Point center(img.cols / 2, img.rows / 2);
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    cv::circle(mask, center, 50, cv::Scalar(255), -1);
    cv::Mat mask_inv;
    cv::bitwise_not(mask, mask_inv);

    cv::Mat high_freq_img_fourier;
    img_fourier.copyTo(high_freq_img_fourier);
    krasivSpektr(high_freq_img_fourier);
    //cv::bitwise_and(img_fourier, img_fourier, low_freq_img_fourier, mask);
    cv::circle(high_freq_img_fourier, center, 50, cv::Scalar(0), -1);
    krasivSpektr(high_freq_img_fourier);

    cv::Mat low_freq_img_fourier = img_fourier - high_freq_img_fourier;

    cv::Mat high_freq_img;
    cv::dft(high_freq_img_fourier, high_freq_img, cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE | cv::DFT_SCALE);
    high_freq_img.convertTo(high_freq_img, CV_8U);

    cv::Mat low_freq_img;
    cv::dft(low_freq_img_fourier, low_freq_img, cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE | cv::DFT_SCALE);
    low_freq_img.convertTo(low_freq_img, CV_8U);


    cv::imshow("mask", mask);
    cv::imshow("high_freq_img_fourier", getPrettyFourier(high_freq_img_fourier));
    cv::imshow("low_freq_img_fourier", getPrettyFourier(low_freq_img_fourier));
    cv::imshow("high_freq_img", high_freq_img);
    cv::imshow("low_freq_img", low_freq_img);
    
    cv::waitKey();

}


void task_6() {
    // Read image in grayscale and convert to double
    std::string image_path = "/home/student2/Pictures/Screenshot from 2023-03-22 17-22-43.png";
    std::string sample_path = "/home/student2/Pictures/Screenshot from 2023-03-22 17-23-02.png";

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat sample = cv::imread(sample_path, cv::IMREAD_GRAYSCALE);

    cv::imshow("img", img);
    cv::imshow("sample", sample);
    cv::waitKey();

    cv::Size orig_img_size(img.cols, img.rows);
    
    img.convertTo(img, CV_64F);
    sample.convertTo(sample, CV_64F);

    PrepareForMullSpectrum(img, sample, img, sample);
    cv::Mat img_fourier;
    cv::dft(img, img_fourier, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat sample_fourier;
    cv::dft(sample, sample_fourier, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat correlation_fourier;
    cv::mulSpectrums(img_fourier, sample_fourier, correlation_fourier, 0, true);

    cv::Mat correlation_img;
    cv::dft(correlation_fourier, correlation_img, cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE | cv::DFT_SCALE);
    correlation_img = correlation_img(cv::Rect(0, 0, orig_img_size.width, orig_img_size.height));
    cv::normalize(correlation_img, correlation_img, 0, 1, cv::NORM_MINMAX);

    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;
    minMaxLoc(correlation_img, &minVal, &maxVal, &minLoc, &maxLoc);

    double corr_thresh = maxVal - 0.01;
    cv::Mat threshold_img;
    cv::threshold(correlation_img, threshold_img, corr_thresh, 255, cv::THRESH_BINARY);

    cv::imshow("img_fourier", getPrettyFourier(img_fourier));
    cv::imshow("filter_fourier", getPrettyFourier(sample_fourier));
    cv::imshow("filtered_img_fourier", getPrettyFourier(correlation_fourier));
    cv::imshow("filtered_img", correlation_img);
    cv::imshow("threshold_img", threshold_img);
    cv::waitKey();
}

void mulSpectrumShow(cv::InputArray orig_img, cv::InputArray filter) {
    
    cv::Size orig_img_size(orig_img.cols(), orig_img.rows());
    cv::Mat img, filter_mat;
    PrepareForMullSpectrum(orig_img, filter, img, filter_mat);

    cv::Mat img_fourier;
    cv::dft(img, img_fourier, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat filter_fourier;
    cv::dft(filter_mat, filter_fourier, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat filtered_img_fourier;
    cv::mulSpectrums(img_fourier, filter_fourier, filtered_img_fourier, 0);

    cv::Mat filtered_img;
    cv::dft(filtered_img_fourier, filtered_img, cv::DFT_REAL_OUTPUT | cv::DFT_INVERSE | cv::DFT_SCALE);
    filtered_img = filtered_img(cv::Rect(0, 0, orig_img_size.width, orig_img_size.height));
    filtered_img.convertTo(filtered_img, CV_8U);

    cv::imshow("img_fourier", getPrettyFourier(img_fourier));
    cv::imshow("filter_fourier", getPrettyFourier(filter_fourier));
    cv::imshow("filtered_img_fourier", getPrettyFourier(filtered_img_fourier));
    cv::imshow("filtered_img", filtered_img);
    cv::waitKey();
}


void test_dft() {
    uint8_t img_data[] = {0, 100, 0, 100, 
                          100, 0, 100, 0,
                          0, 100, 0, 100,
                          100, 100, 100, 100,};
    cv::Mat img(4, 4, CV_8U, img_data);
    img.convertTo(img, CV_64F);
        
    cv::Mat my_fourier, opencv_fourier;
    
    dft2D(img, my_fourier, DFT_COMPLEX_OUTPUT);
    // dft2D(fourier, img_result, DFT_INVERSE | DFT_REAL_OUTPUT);
    
    cv::dft(img, opencv_fourier, cv::DFT_COMPLEX_OUTPUT);
    // cv::dft(fourier, img_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    std::cout << "My\n";
    printMat<std::complex<double>>(my_fourier);
    std::cout << "OpenCV\n";
    printMat<std::complex<double>>(opencv_fourier);

    cv::Mat my_preatty_fourier = getPrettyFourier(my_fourier);
    cv::Mat opencv_preatty_fourier = getPrettyFourier(opencv_fourier);

    cv::Size window_size(200, 200);
    cv::Mat img_resized, my_preatty_fourier_resized, opencv_preatty_fourier_resized;
    cv::resize(img, img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(my_preatty_fourier, my_preatty_fourier_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(opencv_preatty_fourier, opencv_preatty_fourier_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    
    cv::imshow("img", img_resized);
    cv::imshow("my_preatty_fourier", my_preatty_fourier_resized);
    cv::imshow("opencv_preatty_fourier", opencv_preatty_fourier_resized);

    cv::waitKey();
}


void test_idft() {
    uint8_t img_data[] = {0, 100, 0, 100, 
                          100, 0, 100, 0,
                          0, 100, 0, 100,
                          100, 100, 100, 100,};
    cv::Mat img(4, 4, CV_8U, img_data);
    img.convertTo(img, CV_64F);
        
    cv::Mat my_fourier, opencv_fourier;
    cv::Mat my_img, opencv_img;
    
    dft2D(img, my_fourier, DFT_COMPLEX_OUTPUT);
    dft2D(my_fourier, my_img, DFT_INVERSE | DFT_REAL_OUTPUT);
    
    cv::dft(img, opencv_fourier, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(opencv_fourier, opencv_img, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    std::cout << "My\n";
    printMat<double>(my_img);
    std::cout << "OpenCV\n";
    printMat<double>(opencv_img);

    img.convertTo(img, CV_8U);
    my_img.convertTo(my_img, CV_8U);
    opencv_img.convertTo(opencv_img, CV_8U);

    cv::Size window_size(200, 200);
    cv::Mat img_resized, my_img_resized, opencv_img_resized;
    cv::resize(img, img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(my_img, my_img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(opencv_img, opencv_img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    
    cv::imshow("img", img_resized);
    cv::imshow("my_img", my_img_resized);
    cv::imshow("opencv_img", opencv_img_resized);

    cv::waitKey();
}


void test_fft() {
        uint8_t img_data[] = {0, 100, 0, 100, 
                          100, 0, 100, 0,
                          0, 100, 0, 100,
                          100, 100, 100, 100,};
    cv::Mat img(4, 3, CV_8U, img_data);
    img.convertTo(img, CV_64F);
        
    cv::Mat my_fourier, opencv_fourier;
    
    fft2D(img, my_fourier, DFT_COMPLEX_OUTPUT);
    // dft2D(fourier, img_result, DFT_INVERSE | DFT_REAL_OUTPUT);
    
    cv::dft(img, opencv_fourier, cv::DFT_COMPLEX_OUTPUT);
    // cv::dft(fourier, img_result, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    std::cout << "My\n";
    printMat<std::complex<double>>(my_fourier);
    std::cout << "OpenCV\n";
    printMat<std::complex<double>>(opencv_fourier);

    cv::Mat my_preatty_fourier = getPrettyFourier(my_fourier);
    cv::Mat opencv_preatty_fourier = getPrettyFourier(opencv_fourier);

    cv::Size window_size(200, 200);
    cv::Mat img_resized, my_preatty_fourier_resized, opencv_preatty_fourier_resized;
    cv::resize(img, img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(my_preatty_fourier, my_preatty_fourier_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(opencv_preatty_fourier, opencv_preatty_fourier_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    
    cv::imshow("img", img_resized);
    cv::imshow("my_preatty_fourier", my_preatty_fourier_resized);
    cv::imshow("opencv_preatty_fourier", opencv_preatty_fourier_resized);

    cv::waitKey();
}


void test_ifft() {
    uint8_t img_data[] = {0, 100, 0, 100, 
                          100, 0, 100, 0,
                          0, 100, 0, 100,
                          100, 100, 100, 100,};
    cv::Mat img(4, 3, CV_8U, img_data);
    img.convertTo(img, CV_64F);
        
    cv::Mat my_fourier, opencv_fourier;
    cv::Mat my_img, opencv_img;
    
    fft2D(img, my_fourier, DFT_COMPLEX_OUTPUT);
    fft2D(my_fourier, my_img, DFT_INVERSE | DFT_REAL_OUTPUT);
    
    cv::dft(img, opencv_fourier, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(opencv_fourier, opencv_img, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    std::cout << "My\n";
    printMat<double>(my_img);
    std::cout << "OpenCV\n";
    printMat<double>(opencv_img);

    img.convertTo(img, CV_8U);
    my_img.convertTo(my_img, CV_8U);
    opencv_img.convertTo(opencv_img, CV_8U);

    cv::Size window_size(200, 200);
    cv::Mat img_resized, my_img_resized, opencv_img_resized;
    cv::resize(img, img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(my_img, my_img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(opencv_img, opencv_img_resized, window_size, 0.0, 0.0, cv::INTER_NEAREST);
    
    cv::imshow("img", img_resized);
    cv::imshow("my_img", my_img_resized);
    cv::imshow("opencv_img", opencv_img_resized);

    cv::waitKey();
}


void krasivSpektr(cv::Mat &magI){
	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}


void dft1D(cv::InputArray src, cv::OutputArray dst, int flag, cv::InputArray w) {
    
    // Input array must be a 1d-array
    if (src.rows() != 1 && src.cols() != 1) {
        throw cv::Exception();
    }
    int m = std::max(src.rows(), src.cols());
    
    // Use inverse dft instead of forward
    bool invert = flag & (1 << 0);

    // Calculation of constants
    cv::Mat w_mat = w.getMat();
    if (w_mat.empty()) {
        int sign = (invert) ? 1 : -1;
        calc_dft_coef(m, sign, w_mat);
    }
    
    cv::Mat src_mat = src.getMat().clone();
    cv::Mat zero_mat = cv::Mat::zeros(cv::Size(src.cols(), src.rows()), src.type());

    cv::Mat complex_src_mat;

    if (src_mat.channels() == 1) {
        std::vector<cv::Mat> mv = {src_mat, zero_mat};
        cv::merge(mv, complex_src_mat);
    } else {
        complex_src_mat = src_mat;
    }

    if (src.rows() == 1) {
        cv::transpose(complex_src_mat, complex_src_mat);
    }

    cv::Mat dst_mat = w_mat * complex_src_mat;
    
    if (invert) {
        dst_mat /= m;
    }

    if (src.rows() == 1) {
        cv::transpose(dst_mat, dst_mat);
    }

    bool output_is_real = flag & (1 << 1);
    
    if (output_is_real) {
        std::vector<cv::Mat> dst_mats(2);
        cv::split(dst_mat, dst_mats);
        dst_mats[0].copyTo(dst);
        return;    
    }

    dst_mat.copyTo(dst);
}

void dft2D(cv::InputArray src, cv::OutputArray dst, int flag) {
    cv::Mat src_mat = src.getMat();
    cv::Mat dst_mat(src.rows(), src.cols(), CV_64FC2);

    bool invert = flag & (1 << 0);
    int sign = (invert) ? 1 : -1;
    bool output_is_real = flag & (1 << 1);
    flag &= ~(1 << 1);

    // rows
    cv::Mat w_row_mat;
    calc_dft_coef(src.cols(), sign, w_row_mat);
    for (int row = 0; row < src.rows(); row++) {
        cv::Mat src_mat_roi = src_mat(cv::Range(row, row + 1), cv::Range(0, src.cols()));
        cv::Mat dst_mat_roi = dst_mat(cv::Range(row, row + 1), cv::Range(0, src.cols()));
        dft1D(src_mat_roi, dst_mat_roi, flag, w_row_mat);
    }

    // cols
    cv::Mat w_col_mat;
    calc_dft_coef(src.rows(), sign, w_col_mat);
    for (int col = 0; col < src.cols(); col++) {
        cv::Mat dst1_mat_roi = dst_mat(cv::Range(0, src.rows()), cv::Range(col, col + 1));
        cv::Mat dst2_mat_roi = dst_mat(cv::Range(0, src.rows()), cv::Range(col, col + 1));
        dft1D(dst1_mat_roi, dst2_mat_roi, flag, w_col_mat);
    }

    if (output_is_real) {
        cv::Mat dst_mat_real, dst_mat_imag;
        std::vector<cv::Mat> channels = {dst_mat_real, dst_mat_imag};
        cv::split(dst_mat, channels);
        dst_mat = channels[0];
    }

    dst_mat.copyTo(dst);
}


int rev(int num, int lg_n) {
	int res = 0;
	for (int i=0; i < lg_n; i++)
		if (num & (1 << i))
			res |= 1 << (lg_n - 1 - i);
	return res;
}


void fft1D(cv::InputArray src, cv::OutputArray dst, int flag) {
    
    // Input array must be a 1d-array
    if (src.rows() != 1 && src.cols() != 1) {
        throw cv::Exception();
    }

    cv::Mat src_mat = src.getMat();
    int rows = cv::getOptimalDFTSize(src.rows());
    int cols = cv::getOptimalDFTSize(src.cols());

    cv::Mat dst_mat;
    if (src_mat.channels() == 1) {
        std::vector<cv::Mat> mv = {src_mat, cv::Mat::zeros(src_mat.rows, src_mat.cols, CV_64F)};
        cv::merge(mv, dst_mat);
    } else {
        dst_mat = src_mat;
    }

    dst_mat.create(rows, cols, dst_mat.type());
    
    if (src.rows() == 1) {
        cv::transpose(dst_mat, dst_mat);
    }

    int n = std::max(src.rows(), src.cols());
    
    // Number of meaningful bits 
    int lg_n = 0;
	while ((1 << lg_n) < n) lg_n++;
    
    // Bit-reverse permutation
	for (int i = 0; i < n; i++) {
        int reversed_i = rev(i, lg_n);
        if (i < reversed_i) {
            std::swap(dst_mat.ptr<std::complex<double>>(i)[0],
                      dst_mat.ptr<std::complex<double>>(reversed_i)[0]);
        }
    }

    // Use inverse dft instead of forward
    bool invert = flag & (1 << 0);

    // For each stage with "len" elems in interval do butterfly operation
    for (int len = 2; len <= n; len <<= 1) {
		double ang = 2 * M_PI / len * (invert ? 1 : -1);
		std::complex<double> wlen (cos(ang), sin(ang));
		
        // i - start of current interval
        for (int i = 0; i < n; i += len) {
			std::complex<double> w (1);

            // j - index of the first elems in the interval
        	for (int j = 0; j < len / 2; j++) {

				std::complex<double> u = dst_mat.ptr<std::complex<double>>(i + j)[0];
                std::complex<double> v = dst_mat.ptr<std::complex<double>>(i + j + len / 2)[0] * w;
				
                dst_mat.ptr<std::complex<double>>(i + j)[0] = u + v;
				dst_mat.ptr<std::complex<double>>(i + j + len / 2)[0] = u - v;
				
                w *= wlen;
			}
		}
	}

    if (invert) {
        dst_mat /= n;
    }

    if (src.rows() == 1) {
        cv::transpose(dst_mat, dst_mat);
    }

    bool output_is_real = flag & (1 << 1);
    
    if (output_is_real) {
        std::vector<cv::Mat> dst_mats(2);
        cv::split(dst_mat, dst_mats);
        dst_mats[0].copyTo(dst);
        return;    
    }

    dst_mat.copyTo(dst);
}


void fft2D(cv::InputArray src, cv::OutputArray dst, int flag) {
    int new_rows = get_optimal_size(src.rows());
    int new_cols = get_optimal_size(src.cols());
    
    cv::Mat src_mat = cv::Mat::zeros(new_rows, new_cols, src.type());
    cv::Mat orig_src_mat = src.getMat();
    orig_src_mat.copyTo(src_mat(cv::Range(0, src.rows()), cv::Range(0, src.cols())));
    
    cv::Mat dst_mat(src_mat.rows, src_mat.cols, CV_64FC2);

    bool output_is_real = flag & (1 << 1);
    flag &= ~(1 << 1);

    // rows
    for (int row = 0; row < src_mat.rows; row++) {
        cv::Mat src_mat_roi = src_mat(cv::Range(row, row + 1), cv::Range(0, src_mat.cols));
        cv::Mat dst_mat_roi = dst_mat(cv::Range(row, row + 1), cv::Range(0, src_mat.cols));
        fft1D(src_mat_roi, dst_mat_roi, flag);
    }

    // cols
    for (int col = 0; col < src_mat.cols; col++) {
        cv::Mat dst1_mat_roi = dst_mat(cv::Range(0, src_mat.rows), cv::Range(col, col + 1));
        cv::Mat dst2_mat_roi = dst_mat(cv::Range(0, src_mat.rows), cv::Range(col, col + 1));
        fft1D(dst1_mat_roi, dst2_mat_roi, flag);
    }

    if (output_is_real) {
        cv::Mat dst_mat_real, dst_mat_imag;
        std::vector<cv::Mat> channels = {dst_mat_real, dst_mat_imag};
        cv::split(dst_mat, channels);
        dst_mat = channels[0];
    }

    dst_mat.copyTo(dst);
}


void convolveDFT(cv::InputArray A, cv::InputArray B, cv::OutputArray C) {
    
    // reallocate the output array if needed
    C.create(abs(A.rows() - B.rows()) + 1, abs(A.cols() - B.cols()) + 1, A.type());
    cv::Size dftSize;
    
    // calculate the size of DFT transform
    dftSize.width = cv::getOptimalDFTSize(A.cols() + B.cols() - 1);
    dftSize.height = cv::getOptimalDFTSize(A.rows() + B.rows() - 1);
    
    // allocate temporary buffers and initialize them with 0's
    cv::Mat tempA(dftSize, A.type(), cv::Scalar::all(0));
    cv::Mat tempB(dftSize, B.type(), cv::Scalar::all(0));
    
    // copy A and B to the top-left corners of tempA and tempB, respectively
    cv::Mat roiA(tempA, cv::Rect(0, 0, A.cols(), A.rows()));
    A.copyTo(roiA);
    cv::Mat roiB(tempB, cv::Rect(0, 0, B.cols(), B.rows()));
    B.copyTo(roiB);
    
    // now transform the padded A & B in-place;
    // use "nonzeroRows" hint for faster processing
    cv::dft(tempA, tempA, 0, A.rows());
    cv::dft(tempB, tempB, 0, B.rows());
    
    // multiply the spectrums;
    // the function handles packed spectrum representations well
    cv::mulSpectrums(tempA, tempB, tempA, 0);
    
    // transform the product back from the frequency domain.
    // Even though all the result rows will be non-zero,
    // you need only the first C.rows of them, and thus you
    // pass nonzeroRows == C.rows
    cv::dft(tempA, tempA, cv::DFT_INVERSE + cv::DFT_SCALE, C.rows());
    
    // now copy the result back to C.
    tempA(cv::Rect(0, 0, C.cols(), C.rows())).copyTo(C);
    
    // all the temporary buffers will be deallocated automatically
}


void PrepareForMullSpectrum(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst1, cv::OutputArray dst2) {
    
    
    // Calculate the size of DFT transform
    cv::Size dftSize;
    dftSize.width = cv::getOptimalDFTSize(src1.cols() + src2.cols() - 1);
    dftSize.height = cv::getOptimalDFTSize(src1.rows() + src2.rows() - 1);
    
    // Allocate temporary buffers and initialize them with 0's
    cv::Mat temp_src1(dftSize, src1.type(), cv::Scalar::all(0));
    cv::Mat temp_src2(dftSize, src2.type(), cv::Scalar::all(0));
    
    // Copy src1 and src2 to the top-left corners of temp_src1 and temp_src2, respectively
    cv::Mat roi_src1 = temp_src1(cv::Rect(0, 0, src1.cols(), src1.rows()));
    src1.copyTo(roi_src1);
    cv::Mat roi_src2(temp_src2, cv::Rect(0, 0, src2.cols(), src2.rows()));
    src2.copyTo(roi_src2);

    // Copy buffers to output arrays
    temp_src1.copyTo(dst1);
    temp_src2.copyTo(dst2);
}


cv::Mat getPrettyFourier(cv::Mat fourier) {

    cv::Mat channels[2];
    split(fourier, channels);

    cv::Mat magn;
    cv::magnitude(channels[0], channels[1], magn);

    magn += cv::Scalar::all(1);
    log(magn, magn);

    normalize(magn, magn, 0, 1, cv::NormTypes::NORM_MINMAX);
    krasivSpektr(magn);

    return magn;    
}


template <typename T>
void printMat(cv::Mat &img) {
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            std::cout << img.at<T>(row, col) << ' ';
        }
        std::cout << '\n';
    }
}


unsigned char reverse(unsigned char b) {
   b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
   b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
   b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
   return b;
}


int get_optimal_size(int size) {
    int bits = 0;
    int tmp_size = size;
    while (tmp_size != 0) {
        tmp_size >>= 1;
        bits++;
    }
    if ((1 << bits - 1) == size) {
        return size;
    }
    return 1 << bits;
}


void calc_dft_coef(int n, int sign, cv::OutputArray w) {
    
    cv::Mat w_mat = cv::Mat_<std::complex<double>>(n, n);
    std::complex<double> w_diff(1, 0);
    std::complex<double> w_step(cos(sign * 2 * M_PI / n), sin(sign * 2 * M_PI / n));    // exp(+-2pi/m)
    
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (col == 0) {
                w_mat.at<std::complex<double>>(row, col) = std::complex<double>(1, 0);    
            } else {
                w_mat.at<std::complex<double>>(row, col) = w_mat.at<std::complex<double>>(row, col - 1) * w_diff;
            }
        }
        w_diff *= w_step;
    }
    w_mat.copyTo(w);
}


double getSimilarityPercentage(cv::Mat &img1, cv::Mat &img2) {

    int num_of_channels = img1.channels();
    int sum_of_matches = 0;

    double eps = 0.001;

    for (int y = 0; y < img1.rows; y++) {
        auto p1 = img1.ptr<double>(y);
        auto p2 = img2.ptr<double>(y);

        for (int x = 0; x < img1.cols; x++) {
            for (int channel = 0; channel < num_of_channels; channel++) {
                auto value1 = p1[x * num_of_channels + channel];
                auto value2 = p2[x * num_of_channels + channel];
                if (value1 - value2 < value1 * eps || value1 == value2) {
                    sum_of_matches++;
                }
            }
        }
    }

    return double(sum_of_matches) / (img1.cols * img1.rows * num_of_channels) * 100.0;
}


void createOptimalSizedMat(cv::InputArray src, cv::OutputArray dst) {
    int new_rows = get_optimal_size(src.rows());
    int new_cols = get_optimal_size(src.cols());
    
    cv::Mat dst_mat = cv::Mat::zeros(new_rows, new_cols, src.type());
    cv::Mat src_mat = src.getMat();
    
    src_mat.copyTo(dst_mat(cv::Range(0, src.rows()), cv::Range(0, src.cols())));
    dst_mat.copyTo(dst);
}

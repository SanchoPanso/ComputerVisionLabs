#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <complex>

#include <opencv2/opencv.hpp>


unsigned char reverse(unsigned char b) {
   b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
   b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
   b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
   return b;
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


void dft1D(cv::InputArray src, cv::OutputArray dst, int flag = 0) {
    if (src.rows() != 1 && src.cols() != 1) {
        throw cv::Exception();
    }
    int m = std::max(src.rows(), src.cols());

    // Calculation of constants
    cv::Mat w_mat = cv::Mat_<std::complex<double>>(m, m);
    std::complex<double> w_diff(1, 0);
    std::complex<double> w_step(cos(-2 * M_PI / m), sin(-2 * M_PI / m)); // exp(-2pi/m)
    
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            if (col == 0) {
                w_mat.ptr<std::complex<double>>(row)[col] = std::complex<double>(1, 0);    
            } else {
                w_mat.ptr<std::complex<double>>(row)[col] = w_mat.ptr<std::complex<double>>(row)[col - 1] * w_diff;
            }
        }
        w_diff *= w_step;
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
    
    if (src.rows() == 1) {
        cv::transpose(dst_mat, dst_mat);
    }

    dst_mat.copyTo(dst);
}

void dft2D(cv::InputArray src, cv::OutputArray dst, int flag = 0) {
    cv::Mat src_mat = src.getMat();
    cv::Mat dst_mat(src.rows(), src.cols(), CV_64FC2);

    // rows
    for (int row = 0; row < src.rows(); row++) {
        cv::Mat src_mat_roi(src_mat, cv::Range(row, row + 1), cv::Range(0, src.cols()));
        cv::Mat dst_mat_roi(dst_mat, cv::Range(row, row + 1), cv::Range(0, src.cols()));
        dft1D(src_mat_roi, dst_mat_roi);
    }

    // cols
    for (int col = 0; col < src.cols(); col++) {
        cv::Mat dst1_mat_roi(dst_mat, cv::Range(0, src.rows()), cv::Range(col, col + 1));
        cv::Mat dst2_mat_roi(dst_mat, cv::Range(0, src.rows()), cv::Range(col, col + 1));
        dft1D(dst1_mat_roi, dst2_mat_roi);
    }

    dst_mat.copyTo(dst);
}


void fft1D(cv::InputArray src, cv::OutputArray dst, int flag = 0) {
    if (src.cols() != 1) {
        throw cv::Exception();
    }
    int m = src.rows();
    cv::Mat src_mat_orig(src.rows(), src.cols(), src.type());
    cv::Mat src_mat(src.rows(), src.cols(), src.type());

    src.copyTo(src_mat_orig);

    int num_of_divisions = (int) ceil(log2(m));

    for (int i = 0; i < m; i++){
        int inverse_index = reverse(i) >> (8 - num_of_divisions);

        if (inverse_index < m){
            auto p_orig = src_mat_orig.ptr<float>(i);
            auto p = src_mat.ptr<float>(inverse_index);
            p[0] = p_orig[0];
        }
    }

    cv::Mat dst_mat_re(src.rows(), src.cols(), src.type());
    cv::Mat dst_mat_im(src.rows(), src.cols(), src.type());
    
    src_mat.copyTo(dst_mat_re);
    src_mat.copyTo(dst_mat_im);

    for (int i = 0; i < num_of_divisions; i++) {
        int num_in_couple = (1 << i);

        for (int pair = 0; pair < m; pair += 2 * num_in_couple) {
            for (int row = pair; row < std::min(pair + num_in_couple, m); row++) {

                int neib_row = row + num_in_couple;

                auto p_re = dst_mat_re.ptr<float>(row);
                auto p_im = dst_mat_im.ptr<float>(row);

                auto neib_p_re = dst_mat_re.ptr<float>(neib_row % m);
                auto neib_p_im = dst_mat_im.ptr<float>(neib_row % m);
                
                float A_re = p_re[0];
                float A_im = p_im[0];

                float B_re = neib_p_re[0];
                float B_im = neib_p_im[0];

                p_re[0] = A_re + B_re * cos(-2 * 3.14f / (2 * num_in_couple) * (pair - row));
                neib_p_re[0] = A_re - B_re * cos(-2 * 3.14f / (2 * num_in_couple) * (pair - row));

                p_im[0] = A_im + B_im * sin(-2 * 3.14f / (2 * num_in_couple) * (pair - row));
                neib_p_im[0] = A_im + B_im * sin(-2 * 3.14f / (2 * num_in_couple) * (pair - row));
            }
        }
    }

    cv::Mat dst_mat;
    std::vector<cv::Mat> mv = {dst_mat_re, dst_mat_im};
    cv::merge(mv, dst_mat);
    dst_mat.copyTo(dst);
}


int main() {
    std::string image_path = "D:\\CodeProjects\\C_CPP_Projects\\ComputerVisionLabs\\cv_lab4\\250px-Fourier2.jpg";
    // cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    
    double img_data[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    cv::Mat img(3, 4, CV_64F, img_data);
    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(100, 300));
    cv::imshow("img", img_resized);
    
    //img.convertTo(img, CV_32F);
    
    cv::Mat fourier;
    dft2D(img, fourier);
    // cv::dft(img, fourier, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat channels[2];
    split(fourier, channels);

    for (int i = 0; i < channels[0].rows; i++) {
        auto p = channels[0].ptr<double>(i);
        double value = (double) p[0];
        std::cout << value << ' ';
    }
    std::cout << '\n';

    cv::Mat magn;
    cv::magnitude(channels[0], channels[1], magn);

    for (int i = 0; i < magn.rows; i++) {
        double value = magn.at<double>(i, 0);
        std::cout << value << ' ';
    }
    std::cout << '\n';

    magn += cv::Scalar::all(1);
    log(magn, magn);

    normalize(magn, magn, 0, 1, cv::NormTypes::NORM_MINMAX);
    krasivSpektr(magn);

    cv::Mat h = magn * 179.0;
    h.convertTo(h, CV_8U);
    cv::Mat s(h.rows, h.cols, CV_8U, 255);
    cv::Mat v(h.rows, h.cols, CV_8U, 255);

    std::vector<cv::Mat> hsv = {h, s, v};
    cv::Mat color_magn;
    cv::merge(hsv, color_magn);

    cv::Mat magn_resized;
    cv::resize(magn, magn_resized, cv::Size(100, 300));
    
    cv::imshow("magn", magn_resized);
    cv::waitKey();
    
    return 0;
}
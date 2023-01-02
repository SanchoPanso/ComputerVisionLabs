#include <opencv2\opencv.hpp>
#include "cvDirectory.h"
using namespace cv;
using namespace std;


int h_min = 0;
int s_min = 0;
int v_min = 0;

int h_max = 180;
int s_max = 255;
int v_max = 255;

int th = 0;
Mat img, thimg, img_hsv;


void proc_img(int, void* user_data){
	Scalar lower(h_min, s_min, v_min);
	Scalar upper(h_max, s_max, v_max);
	inRange(img_hsv, lower, upper, thimg);
	imshow("thimg1", thimg);
}

int main(int argc, char** argv){
	string fn;
	// if (argc>1) fn= argv[1];
	// else 
	fn = "C:\\Users\\HP\\Downloads\\LAB3\\img_zadan\\roboti\\roi_robotov.jpg";
	img = imread(fn, IMREAD_COLOR);
	resize(img, img, Size(400, 400));
	cvtColor(img, img_hsv, COLOR_BGR2HSV);
	imshow(fn, img); //C:\Users\HP\Downloads\LAB3\img_zadan\allababah
	//vector<string> fnms = Directory::GetListFiles("./","*.jpg",false);//�������� ��� ������ ���� jpg � ������� ����� ��� ���������� � ��� ����.

	int th_type = THRESH_BINARY;

	// proc_img(0, &th_type);
	// createTrackbar("th", "thimg", &th, 255, proc_img, &th_type);

	namedWindow("thimg");

	createTrackbar("h_min", "thimg", &h_min, 180, proc_img, &th_type);
	createTrackbar("s_min", "thimg", &s_min, 255, proc_img, &th_type);
	createTrackbar("v_min", "thimg", &v_min, 255, proc_img, &th_type);

	createTrackbar("h_max", "thimg", &h_max, 180, proc_img, &th_type);
	createTrackbar("s_max", "thimg", &s_max, 255, proc_img, &th_type);
	createTrackbar("v_max", "thimg", &v_max, 255, proc_img, &th_type);

	waitKey();
}
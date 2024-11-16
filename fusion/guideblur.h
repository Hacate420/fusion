#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
class guideblur
{

public:
	guideblur();
	~guideblur();
	cv::Mat fastGuidedFilter(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s);
	//实现快速导向滤波，包含参数转变
	Mat guide(Mat src, int r, double eps, int s);

	//普通引导滤波,img为引导图像，dst为滤波图像,r为窗口半径，eps为正则化参数
	Mat guidedfilter(Mat& img, Mat& dst, int r, double esp);
	//实现普通导向滤波，包含参数转变
	Mat guide_normal(Mat src, int r, double eps);

};
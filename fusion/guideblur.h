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
	//ʵ�ֿ��ٵ����˲�����������ת��
	Mat guide(Mat src, int r, double eps, int s);

	//��ͨ�����˲�,imgΪ����ͼ��dstΪ�˲�ͼ��,rΪ���ڰ뾶��epsΪ���򻯲���
	Mat guidedfilter(Mat& img, Mat& dst, int r, double esp);
	//ʵ����ͨ�����˲�����������ת��
	Mat guide_normal(Mat src, int r, double eps);

};
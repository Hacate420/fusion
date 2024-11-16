#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
class EN_CLAHE
{
	public:
	EN_CLAHE();
	~EN_CLAHE();
	// 一维高斯平滑
	void gaussianSmooth(float* arr, int size, float sigma);
	// 计算均方差
	int deviation(Mat& img);
	//强化的CLAHE 限制对比度的自适应直方图均衡化
	// 1. 计算直方图
	// 2. 裁剪和增加操作
	// 3. 计算累积分布直方图
	// 4. 对子块直方图和全局直方图进行融合
	// 5. 一维方向的均值平滑
	// 6. 一维方向的高斯平滑
	// 7. 二维高斯平滑
	// 8. 计算变换后的像素值
	Mat CLAHE_MY(Mat src,  int limit = 4, int Adaptation = 100);
};



#include "guideblur.h"

using namespace std;
using namespace cv;

guideblur::guideblur()
{
}

guideblur::~guideblur()
{
}
//快速导向滤波
//I_org:引导图像, p_org:滤波图像, r:窗口半径, eps:正则化参数, s:下采样因子
//s=1时，为普通导向滤波, s>1时，为快速导向滤波, 速度更快, 但精度略有下降
//一般s=4, 速度提升4倍, 精度下降不明显, 但s越大, 精度下降越明显
cv::Mat guideblur::fastGuidedFilter(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s)
{

	cv::Mat I, _I;
	_I = I_org.clone();
	resize(_I, I, Size(), 1.0 / s, 1.0 / s, 1);//下采样I

	cv::Mat p, _p;

	_p = p_org.clone();
	resize(_p, p, Size(), 1.0 / s, 1.0 / s, 1); //下采样P

	//[hei, wid] = size(I);    
	int hei = I.rows;
	int wid = I.cols;

	r = (2 * r + 1) / s + 1;//因为opencv自带的boxFilter（）中的Size,比如9x9,我们说半径为4   

	//mean_I = boxfilter(I, r) ./ N;    
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(r, r)); //求I 的均值

	//mean_p = boxfilter(p, r) ./ N;    
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_32FC1, cv::Size(r, r)); //求P 的均值

	//mean_Ip = boxfilter(I.*p, r) ./ N;    
	cv::Mat mean_Ip;                                     //求IP的均值
	cv::boxFilter(I.mul(p), mean_Ip, CV_32FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.    
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;    
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;    
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);    //求I的方差

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;       
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;    
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;    
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
	Mat rmean_a;
	resize(mean_a, rmean_a, Size(I_org.cols, I_org.rows), 1);//上采样

	//mean_b = boxfilter(b, r) ./ N;    
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));
	Mat rmean_b;
	resize(mean_b, rmean_b, Size(I_org.cols, I_org.rows), 1);//上采样

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;    
	cv::Mat q = rmean_a.mul(_I) + rmean_b;

	return q;
}

//实现快速导向滤波，包含参数转变
Mat guideblur::guide(Mat src, int r, double eps, int s){
	Mat src_temp, P_src;
	// 将图像转换为浮点型  
	src.convertTo(src_temp, CV_32FC1, 1.0 / 255.0);
	//生成引导图像
	P_src = src_temp.clone();
	//实现快速导向滤波
	Mat dst = fastGuidedFilter(src_temp, P_src, r, eps, s);
	//将灰度值映射到0-255
	dst.convertTo(dst, CV_8UC1, 255.0);
	//dst.convertTo(dst, CV_32FC1, 255.0);
	return dst;
}

//普通引导滤波
Mat guideblur::guidedfilter(Mat& img, Mat& dst, int r, double esp){
	int row = img.rows;
	int col = img.cols;
	img.convertTo(img, CV_32FC1);
	dst.convertTo(dst, CV_32FC1);
	Mat boxResult, mean_I, mean_P, result, mean_IP, cov_IP, mean_II, var_I;
	Mat a, b, mean_a, mean_b;
	boxFilter(Mat::ones(Size(col, row), img.type()), boxResult, CV_32FC1, Size(r, r));//计算均值N
	boxFilter(img, mean_I, CV_32FC1, Size(r, r));
	mean_I = mean_I / boxResult;//计算导向均值mean_I 
	boxFilter(dst, mean_P, CV_32FC1, Size(r, r));
	mean_P = mean_P / boxResult;//计算原始均值mean_P 
	boxFilter(img.mul(dst), mean_IP, CV_32FC1, Size(r, r));
	mean_IP = mean_IP / boxResult;//计算互相关均值mean_IP corrIP
	cov_IP = mean_IP - mean_I.mul(mean_P);
	boxFilter(img.mul(img), mean_II, CV_32FC1, Size(r, r));
	mean_II = mean_II / boxResult;//计算自相关均值mean_II   corrI
	var_I = mean_II - mean_I.mul(mean_I);//计算方差var_I 
	a = cov_IP / (var_I + esp);//计算相关系数 
	b = mean_P - a.mul(mean_I);
	boxFilter(a, mean_a, CV_32FC1, Size(r, r));
	boxFilter(b, mean_b, CV_32FC1, Size(r, r));
	mean_a = mean_a / boxResult;//计算mean_a 
	mean_b = mean_b / boxResult;//计算mean_b
	result = mean_a.mul(img) + mean_b;//计算P
	return result;
}

//实现普通导向滤波，包含参数转变
Mat guideblur::guide_normal(Mat src, int r, double eps) {
	Mat src_temp, P_src;
	// 将图像转换为浮点型  
	src.convertTo(src_temp, CV_32FC1, 1.0 / 255.0);
	//生成引导图像
	P_src = src_temp.clone();
	//实现快速导向滤波
	Mat dst = guidedfilter(src_temp, P_src, r, eps);
	//将灰度值映射到0-255
	dst.convertTo(dst, CV_8UC1, 255.0);
	//dst.convertTo(dst, CV_32FC1, 255.0);
	return dst;
}
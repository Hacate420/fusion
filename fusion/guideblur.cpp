#include "guideblur.h"

using namespace std;
using namespace cv;

guideblur::guideblur()
{
}

guideblur::~guideblur()
{
}
//���ٵ����˲�
//I_org:����ͼ��, p_org:�˲�ͼ��, r:���ڰ뾶, eps:���򻯲���, s:�²�������
//s=1ʱ��Ϊ��ͨ�����˲�, s>1ʱ��Ϊ���ٵ����˲�, �ٶȸ���, �����������½�
//һ��s=4, �ٶ�����4��, �����½�������, ��sԽ��, �����½�Խ����
cv::Mat guideblur::fastGuidedFilter(cv::Mat I_org, cv::Mat p_org, int r, double eps, int s)
{

	cv::Mat I, _I;
	_I = I_org.clone();
	resize(_I, I, Size(), 1.0 / s, 1.0 / s, 1);//�²���I

	cv::Mat p, _p;

	_p = p_org.clone();
	resize(_p, p, Size(), 1.0 / s, 1.0 / s, 1); //�²���P

	//[hei, wid] = size(I);    
	int hei = I.rows;
	int wid = I.cols;

	r = (2 * r + 1) / s + 1;//��Ϊopencv�Դ���boxFilter�����е�Size,����9x9,����˵�뾶Ϊ4   

	//mean_I = boxfilter(I, r) ./ N;    
	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(r, r)); //��I �ľ�ֵ

	//mean_p = boxfilter(p, r) ./ N;    
	cv::Mat mean_p;
	cv::boxFilter(p, mean_p, CV_32FC1, cv::Size(r, r)); //��P �ľ�ֵ

	//mean_Ip = boxfilter(I.*p, r) ./ N;    
	cv::Mat mean_Ip;                                     //��IP�ľ�ֵ
	cv::boxFilter(I.mul(p), mean_Ip, CV_32FC1, cv::Size(r, r));

	//cov_Ip = mean_Ip - mean_I .* mean_p; % this is the covariance of (I, p) in each local patch.    
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

	//mean_II = boxfilter(I.*I, r) ./ N;    
	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(r, r));

	//var_I = mean_II - mean_I .* mean_I;    
	cv::Mat var_I = mean_II - mean_I.mul(mean_I);    //��I�ķ���

	//a = cov_Ip ./ (var_I + eps); % Eqn. (5) in the paper;       
	cv::Mat a = cov_Ip / (var_I + eps);

	//b = mean_p - a .* mean_I; % Eqn. (6) in the paper;    
	cv::Mat b = mean_p - a.mul(mean_I);

	//mean_a = boxfilter(a, r) ./ N;    
	cv::Mat mean_a;
	cv::boxFilter(a, mean_a, CV_32FC1, cv::Size(r, r));
	Mat rmean_a;
	resize(mean_a, rmean_a, Size(I_org.cols, I_org.rows), 1);//�ϲ���

	//mean_b = boxfilter(b, r) ./ N;    
	cv::Mat mean_b;
	cv::boxFilter(b, mean_b, CV_32FC1, cv::Size(r, r));
	Mat rmean_b;
	resize(mean_b, rmean_b, Size(I_org.cols, I_org.rows), 1);//�ϲ���

	//q = mean_a .* I + mean_b; % Eqn. (8) in the paper;    
	cv::Mat q = rmean_a.mul(_I) + rmean_b;

	return q;
}

//ʵ�ֿ��ٵ����˲�����������ת��
Mat guideblur::guide(Mat src, int r, double eps, int s){
	Mat src_temp, P_src;
	// ��ͼ��ת��Ϊ������  
	src.convertTo(src_temp, CV_32FC1, 1.0 / 255.0);
	//��������ͼ��
	P_src = src_temp.clone();
	//ʵ�ֿ��ٵ����˲�
	Mat dst = fastGuidedFilter(src_temp, P_src, r, eps, s);
	//���Ҷ�ֵӳ�䵽0-255
	dst.convertTo(dst, CV_8UC1, 255.0);
	//dst.convertTo(dst, CV_32FC1, 255.0);
	return dst;
}

//��ͨ�����˲�
Mat guideblur::guidedfilter(Mat& img, Mat& dst, int r, double esp){
	int row = img.rows;
	int col = img.cols;
	img.convertTo(img, CV_32FC1);
	dst.convertTo(dst, CV_32FC1);
	Mat boxResult, mean_I, mean_P, result, mean_IP, cov_IP, mean_II, var_I;
	Mat a, b, mean_a, mean_b;
	boxFilter(Mat::ones(Size(col, row), img.type()), boxResult, CV_32FC1, Size(r, r));//�����ֵN
	boxFilter(img, mean_I, CV_32FC1, Size(r, r));
	mean_I = mean_I / boxResult;//���㵼���ֵmean_I 
	boxFilter(dst, mean_P, CV_32FC1, Size(r, r));
	mean_P = mean_P / boxResult;//����ԭʼ��ֵmean_P 
	boxFilter(img.mul(dst), mean_IP, CV_32FC1, Size(r, r));
	mean_IP = mean_IP / boxResult;//���㻥��ؾ�ֵmean_IP corrIP
	cov_IP = mean_IP - mean_I.mul(mean_P);
	boxFilter(img.mul(img), mean_II, CV_32FC1, Size(r, r));
	mean_II = mean_II / boxResult;//��������ؾ�ֵmean_II   corrI
	var_I = mean_II - mean_I.mul(mean_I);//���㷽��var_I 
	a = cov_IP / (var_I + esp);//�������ϵ�� 
	b = mean_P - a.mul(mean_I);
	boxFilter(a, mean_a, CV_32FC1, Size(r, r));
	boxFilter(b, mean_b, CV_32FC1, Size(r, r));
	mean_a = mean_a / boxResult;//����mean_a 
	mean_b = mean_b / boxResult;//����mean_b
	result = mean_a.mul(img) + mean_b;//����P
	return result;
}

//ʵ����ͨ�����˲�����������ת��
Mat guideblur::guide_normal(Mat src, int r, double eps) {
	Mat src_temp, P_src;
	// ��ͼ��ת��Ϊ������  
	src.convertTo(src_temp, CV_32FC1, 1.0 / 255.0);
	//��������ͼ��
	P_src = src_temp.clone();
	//ʵ�ֿ��ٵ����˲�
	Mat dst = guidedfilter(src_temp, P_src, r, eps);
	//���Ҷ�ֵӳ�䵽0-255
	dst.convertTo(dst, CV_8UC1, 255.0);
	//dst.convertTo(dst, CV_32FC1, 255.0);
	return dst;
}
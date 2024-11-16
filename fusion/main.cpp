#include <opencv2/opencv.hpp>
#include "fusion.h"
#include "guideblur.h"
#include "EN_CLAHE.h"

using namespace std;
using namespace cv;


//融合图像
int main()
{
	


	/*Mat dop = imread("F:\\偏振成像\\实验数据\\8\\dop-均匀.png");
	Mat aop = imread("F:\\偏振成像\\实验数据\\8\\aop-均匀.png");
	Mat ori = imread("F:\\偏振成像\\实验数据\\8\\可见光-均匀.png");*/

	
	//融合前预处理
	Mat dop = imread("F:\\偏振成像\\实验数据\\8\\dop-均匀.png",0);
	Mat aop = imread("F:\\偏振成像\\实验数据\\8\\aop-均匀.png",0);
	Mat ori = imread("F:\\偏振成像\\实验数据\\8\\kjg-均匀.png",0);
	Mat img = imread("F:\\偏振成像\\测试代码\\fusion\\fusion\\1.jpg",0);

	// 计算直方图
	cv::Mat hist;
	const int channels[] = { 0 };
	const int histSize[] = { 256 }; // 因为是灰度图像，所以有256个级别
	float hranges[] = { 0, 256 };
	const float* ranges[] = { hranges };
	cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
	// 计算均值和总和
	double meanBrightness = 0.0;
	for (int i = 0; i < 256; ++i) {
		meanBrightness += i * hist.at<float>(i); // 这里假设hist是以浮点数存储的
	}
	meanBrightness /= img.total(); // 总和除以像素总数得到均值
	// 计算标准差
	double variance = 0.0;
	for (int i = 0; i < 256; ++i) {
		variance += hist.at<float>(i) * std::pow((i - meanBrightness), 2);
	}
	double standardDeviation = std::sqrt(variance / img.total());
	std::cout << "Standard deviation of brightness: " << standardDeviation << std::endl;

	//guideblur GuideBlur;
	//EN_CLAHE Clahe;

	//Mat dop_guide = GuideBlur.guide(dop, 5, 0.0001, 4);
	//Mat aop_guide = GuideBlur.guide(aop, 5, 0.0001, 4);
	//Mat ori_guide = GuideBlur.guide(ori, 5, 0.0001, 4);

	//Mat dop_cla = Clahe.CLAHE_MY(dop_guide);
	//Mat aop_cla = Clahe.CLAHE_MY(aop_guide);
	//Mat ori_cla = Clahe.CLAHE_MY(ori_guide);

	///*imshow("dop_cla", dop_cla);
	//imshow("aop_cla", aop_cla);
	//imshow("ori_cla", ori_cla);
	//waitKey(0);
	//imwrite("9_dop_cla.jpg", dop_cla);
	//imwrite("9_aop_cla.jpg", aop_cla);
	//imwrite("9_ori_cla.jpg", ori_cla);*/


	////融合
	//vector<Mat> src_arr;
	//src_arr.push_back(dop);
	//src_arr.push_back(aop);
	//src_arr.push_back(ori);


	//fusion fusion;
	//Mat out = fusion.Run(src_arr);

	//imshow("out", out);
	////imwrite("out.jpg", out);
	//waitKey(0);

	return 0;
}

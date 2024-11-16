#include <opencv2/opencv.hpp>
#include "fusion.h"
#include "guideblur.h"
#include "EN_CLAHE.h"

using namespace std;
using namespace cv;


//�ں�ͼ��
int main()
{
	


	/*Mat dop = imread("F:\\ƫ�����\\ʵ������\\8\\dop-����.png");
	Mat aop = imread("F:\\ƫ�����\\ʵ������\\8\\aop-����.png");
	Mat ori = imread("F:\\ƫ�����\\ʵ������\\8\\�ɼ���-����.png");*/

	
	//�ں�ǰԤ����
	Mat dop = imread("F:\\ƫ�����\\ʵ������\\8\\dop-����.png",0);
	Mat aop = imread("F:\\ƫ�����\\ʵ������\\8\\aop-����.png",0);
	Mat ori = imread("F:\\ƫ�����\\ʵ������\\8\\kjg-����.png",0);
	Mat img = imread("F:\\ƫ�����\\���Դ���\\fusion\\fusion\\1.jpg",0);

	// ����ֱ��ͼ
	cv::Mat hist;
	const int channels[] = { 0 };
	const int histSize[] = { 256 }; // ��Ϊ�ǻҶ�ͼ��������256������
	float hranges[] = { 0, 256 };
	const float* ranges[] = { hranges };
	cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
	// �����ֵ���ܺ�
	double meanBrightness = 0.0;
	for (int i = 0; i < 256; ++i) {
		meanBrightness += i * hist.at<float>(i); // �������hist���Ը������洢��
	}
	meanBrightness /= img.total(); // �ܺͳ������������õ���ֵ
	// �����׼��
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


	////�ں�
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

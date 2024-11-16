#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
class EN_CLAHE
{
	public:
	EN_CLAHE();
	~EN_CLAHE();
	// һά��˹ƽ��
	void gaussianSmooth(float* arr, int size, float sigma);
	// ���������
	int deviation(Mat& img);
	//ǿ����CLAHE ���ƶԱȶȵ�����Ӧֱ��ͼ���⻯
	// 1. ����ֱ��ͼ
	// 2. �ü������Ӳ���
	// 3. �����ۻ��ֲ�ֱ��ͼ
	// 4. ���ӿ�ֱ��ͼ��ȫ��ֱ��ͼ�����ں�
	// 5. һά����ľ�ֵƽ��
	// 6. һά����ĸ�˹ƽ��
	// 7. ��ά��˹ƽ��
	// 8. ����任�������ֵ
	Mat CLAHE_MY(Mat src,  int limit = 4, int Adaptation = 100);
};



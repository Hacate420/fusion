#include "EN_CLAHE.h"

EN_CLAHE::EN_CLAHE()
{
	
}
EN_CLAHE::~EN_CLAHE()
{

}

//���������
int EN_CLAHE::deviation(Mat& img) {
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
	return standardDeviation;
}


// һά��˹ƽ��
void EN_CLAHE::gaussianSmooth(float* arr, int size, float sigma) {
	float* smoothed = new float[size];

	// �����˹��
	float* kernel = new float[size];
	float sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		kernel[i] = exp(-(i - size / 2) * (i - size / 2) / (2 * sigma * sigma));
		sum += kernel[i];
	}
	// Normalize the kernel
	for (int i = 0; i < size; ++i) {
		kernel[i] /= sum;
	}

	// ���о��
	for (int i = 0; i < size; ++i) {
		float sum = 0.0f;
		for (int j = 0; j < size; ++j) {
			int index = (i - size / 2 + j + size) % size; // Handle circular convolution
			sum += arr[index] * kernel[j];
		}
		smoothed[i] = sum;
	}

	// ��ƽ�����ֵ���ƻ�ԭʼ����
	for (int i = 0; i < size; ++i) {
		arr[i] = smoothed[i];
	}

	delete[] smoothed;
	delete[] kernel;
}

//CLAHE ���ƶԱȶȵ�����Ӧֱ��ͼ���⻯
// 1. ����ֱ��ͼ
// 2. �ü������Ӳ���
// 3. �����ۻ��ֲ�ֱ��ͼ
// 4. ���ӿ�ֱ��ͼ��ȫ��ֱ��ͼ�����ں�
// 5. һά����ľ�ֵƽ��
// 6. һά����ĸ�˹ƽ��
// 7. ��ά��˹ƽ��
// 8. ����任�������ֵ
Mat EN_CLAHE::CLAHE_MY(Mat src, int limit, int Adaptation )
{
	Mat CLAHE_GO = src.clone();
	int width = src.cols;
	int height = src.rows;
	int deviation_num = deviation(src);
	int block;
	if (deviation_num < 20) {
		block = 8;
	}
	else {
		block = 4;
	}
	int width_block = width / block; //ÿ��С���ӵĳ��Ϳ�
	int height_block = height / block;
	//�洢����ֱ��ͼ  
	int(*tmp2)[256] = new int[block * block][256]{ 0 };
	float(*C2)[256] = new float[block * block][256]{ 0.0 };
	// ��ȡȫͼ������ֱ��ͼ
	Mat histL;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), histL, 1, &histSize, &histRange);

	//�ֿ�
	int total = width_block * height_block;
	for (int i = 0; i < block; i++)
	{
		for (int j = 0; j < block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//����С��,����ֱ��ͼ
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}
			//�ü������Ӳ���
			int LIMIT = limit * width_block * height_block / 255;
			int steal = 0;
			for (int k = 0; k < 256; k++)
			{
				if (tmp2[num][k] > LIMIT) {
					steal += tmp2[num][k] - LIMIT;
					tmp2[num][k] = LIMIT;
				}
			}
			int bonus = steal / 256;
			//hand out the steals averagely�����������ص������Ҷȼ�
			for (int k = 0; k < 256; k++)
			{
				tmp2[num][k] += bonus;
			}

			// ���ӿ�ֱ��ͼ��ȫ��ֱ��ͼ�����ں�
			for (int i = 0; i < block * block; i++)
			{
				for (int j = 0; j < 256; j++)
				{
					C2[i][j] = (C2[i][j] * Adaptation + (100 - Adaptation) * histL.at<float>(j)) / 100;
				}
			}

			//�����ۻ��ֲ�ֱ��ͼ  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0) {
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				}
				else {
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
				}
			}




			// ��ȫ��һά����ľ�ֵƽ��
		   // ˮƽ����
			//const int smoothFactor = 3; // ��ֵƽ���Ĵ��ڴ�С
			/*for (int k = 0; k < 256; k++) {
				float sum = 0.0f;
				int count = 0;
				for (int ii = k - smoothFactor; ii <= k + smoothFactor; ii++) {
					if (ii >= 0 && ii < 256) {
						sum += C2[num][ii];
						count++;
					}
				}
				C2[num][k] = sum / count;
			}*/
			// ��ȫ��һά����ĸ�˹ƽ��
		   // ˮƽ����
			//const int smoothFactor = 3; // ��˹ƽ���Ĵ��ڴ�С
			//float* smoothedCDF = new float[256] { 0.0 };
			//for (int k = 0; k < 256; k++)
			//{
			//	float sum = 0.0f;
			//	float weights = 0.0f;
			//	for (int ii = k - smoothFactor; ii <= k + smoothFactor; ii++)
			//	{
			//		if (ii >= 0 && ii < 256)
			//		{
			//			float weight = exp(-(ii - k) * (ii - k) / (2 * smoothFactor * smoothFactor));
			//			sum += C2[num][ii] * weight;
			//			weights += weight;
			//		}
			//	}
			//	smoothedCDF[k] = sum / weights;
			//}

			//// ��ƽ������ۻ��ֲ�ֱ��ͼ���C2����
			//for (int k = 0; k < 256; k++)
			//{
			//	C2[num][k] = smoothedCDF[k];
			//}

			//delete[] smoothedCDF;

			 //��ÿ���ӿ��ӳ���Ӧ��һά��˹ƽ��
			float sigma = 1.0f; // You can adjust the value of sigma as needed
			int kernel_size = 9; // You can adjust the kernel size as needed
			for (int i = 0; i < block * block; ++i) {
				gaussianSmooth(C2[i], 256, sigma);
			}

			// ��ȫ�ֶ�ά��˹ƽ��
			//Mat gaussian_kernel = getGaussianKernel(5, 1.0, CV_32F); // 5x5 Gaussian kernel with sigma = 1.0
			//Mat mapping_table(1, 256, CV_32F, C2[num]);
			//filter2D(mapping_table, mapping_table, -1, gaussian_kernel, Point(-1, -1), 0, BORDER_DEFAULT);

		}
	}
	//����任�������ֵ  
	//�������ص��λ�ã�ѡ��ͬ�ļ��㷽��  
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			//four coners  
			if (i <= width_block / 2 && j <= height_block / 2)
			{
				int num = 0;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i <= width_block / 2 && j >= ((block - 1) * height_block + height_block / 2)) {
				int num = block * (block - 1);
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1) * width_block + width_block / 2) && j <= height_block / 2) {
				int num = block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1) * width_block + width_block / 2) && j >= ((block - 1) * height_block + height_block / 2)) {
				int num = block * block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			//four edges except coners  
			else if (i <= width_block / 2)
			{
				//���Բ�ֵ  
				int num_i = 0;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j * height_block + height_block / 2)) / (1.0f * height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (i >= ((block - 1) * width_block + width_block / 2)) {
				//���Բ�ֵ  
				int num_i = block - 1;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j * height_block + height_block / 2)) / (1.0f * height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j <= height_block / 2) {
				//���Բ�ֵ  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = 0;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i * width_block + width_block / 2)) / (1.0f * width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j >= ((block - 1) * height_block + height_block / 2)) {
				//���Բ�ֵ  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = block - 1;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i * width_block + width_block / 2)) / (1.0f * width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q * C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//˫���Բ�ֵ
			else {
				int num_i = (i - width_block / 2) / width_block;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				int num3 = num1 + block;
				int num4 = num2 + block;
				float u = (i - (num_i * width_block + width_block / 2)) / (1.0f * width_block);
				float v = (j - (num_j * height_block + height_block / 2)) / (1.0f * height_block);
				CLAHE_GO.at<uchar>(j, i) = (int)((u * v * C2[num4][src.at<uchar>(j, i)] +
					(1 - v) * (1 - u) * C2[num1][src.at<uchar>(j, i)] +
					u * (1 - v) * C2[num2][src.at<uchar>(j, i)] +
					v * (1 - u) * C2[num3][src.at<uchar>(j, i)]) * 255);
			}
		}
	}
	return CLAHE_GO;
}

#include "fusion.h"

using namespace cv;
using namespace std;
guideblur gb;
fusion::fusion()
{
}
fusion::~fusion()
{
}

// ��ͼ��ĺ���
void sharpenImage(Mat image)
{
    //Mat sharpened;
    // ʹ��������˹�˲�������ͼ����
    Mat kernel = (Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    filter2D(image, image, image.depth(), kernel);
    //return sharpened;
}

//�����ݶ���
void grad_sharp(Mat image, Mat dst)
{
    // TODO: �ڴ���ӿؼ�֪ͨ����������
    Mat result, grad;

    result.create(image.size(), CV_8UC1);
    //grad.create(image.size(), CV_8UC1);

    //��Ϊ���һ�����һ��û�а취�����ݶ����ԣ�ʹ��ԭͼ��䣬�����ûҶ�ͼ���г�ʼ��
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            result.at<uchar>(i, j) = image.at<char>(i, j);
        }
    }

    for (int i = 1; i < image.rows - 1; i++)
    {
        for (int j = 1; j < image.cols - 1; j++)
        {
            //grad.at<uchar>(i, j) = saturate_cast<uchar>(fabs(image.at<uchar>(i, j) - image.at<uchar>(i - 1, j)));//+ fabs(image_gray.at<uchar>(i, j) - image_gray.at<uchar>(i , j-1))
            result.at<uchar>(i, j) = saturate_cast<uchar>(image.at<uchar>(i, j) - fabs(image.at<uchar>(i, j) - image.at<uchar>(i - 1, j)));

        }

    }
    dst = result;
}



//����ͼ������Ȩ��
Mat fusion::LightCalculate(Mat src)
{
	Mat light_weight = Mat::zeros(src.size(), CV_32FC1);
	float lut[256];
    //������ұ�lut��һ�����飬lut[i]��һ��ֵ��i������ͼ�������ֵ��lut[i]�����ͼ�������ֵ
    //ʹ��ѭ������0��255��ÿһ������ֵ�����������Ӧ������Ȩ��
    for (int i = 0; i < 256; i++)
    {
        //������ֵ��һ��Ϊ��Χ[0, 1]��Ȼ���ȥ0.5�����������Ŀ�Ŀ�����ʹ������ֵ��0Ϊ����
		float value = pow(i / 255.0 - 0.5, 2);
        //0.2��Ϊ��˹�����ķ��ʹ��ͼ������ȱ仯����ƽ��
        //�������ƺ����Ķ��ͳ̶ȣ�������Ȩ�ر仯�����ʺͷ�Χ
		lut[i] = exp(-value / (2 * (0.2 * 0.2)));
	}

    Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
            //lut:���ұ�,src_gray.at<uchar>(i, j):����ͼ��,light_weight.at<float>(i, j):���ͼ��
			light_weight.at<float>(i, j) = lut[src_gray.at<uchar>(i, j)];
		}
	}
	return light_weight;
}

vector<Mat> fusion::WeightCalculate(vector<Mat> src_arr) {
    vector<Mat> weight_arr;

    for (int i = 0; i < src_arr.size(); i++) {
        Mat light_weight = LightCalculate(src_arr[i]);
        weight_arr.push_back(light_weight);
        light_weight.convertTo(light_weight, CV_8UC1, 255, 0);
        imwrite(format(".\\light_weight_%d.bmp", i), light_weight);
    }

    return weight_arr;
}
//��˹������
vector<Mat> fusion::GaussianPyramid(Mat img, int level) {
    vector<Mat> pyr;
    Mat item = img;
    pyr.push_back(item);
    for (int i = 1; i < level; i++) {
        resize(item, item, item.size() / 2, 0, 0, cv::INTER_AREA);
        //imwrite(format(".\\gaussian_%d.bmp", i), item);
        pyr.push_back(item);
    }
    return pyr;
}

//������˹������
vector<Mat> fusion::LaplacianPyramid(Mat img, int level) {
    vector<Mat> pyr;
    Mat item = img;
    for (int i = 1; i < level; i++) {
        Mat item_down;
        Mat item_up;
        //�²������õ��ͷֱ���ͼ��
        resize(item, item_down, item.size() / 2, 0, 0, INTER_AREA);
        //�ϲ�����ԭʼ item �ĳߴ磬���洢�� item_up ��
        resize(item_down, item_up, item.size());

        //������˹��������ÿһ��Ĳв�
        Mat diff(item.size(), CV_16SC1);
        for (int m = 0; m < item.rows; m++) {
            short* ptr_diff = diff.ptr<short>(m);
            uchar* ptr_up = item_up.ptr(m);
            uchar* ptr_item = item.ptr(m);

            for (int n = 0; n < item.cols; n++) {
                ptr_diff[n] = (short)ptr_item[n] - (short)ptr_up[n];//��в�
            }
        }
        pyr.push_back(diff);
        item = item_down;
        //imwrite(format(".\\laplacian_%d.bmp", i), diff);
    }
    //���ԭʼͼ�񣬵���������ĩβ
    item.convertTo(item, CV_16SC1);
    pyr.push_back(item);

    return pyr;
}

//��Ե�ں�
//��������˹��������ÿһ��Ĳв��Ȩ��ͣ��õ��µĲвȻ���ع�
//pyr_i_arr:Ҫ�ں�ͼ���������˹������,pyr_w_arr:Ȩ�ؽ���������˹������
vector<Mat> fusion::EdgeFusion(vector<vector<Mat>> pyr_i_arr, vector<vector<Mat>> pyr_w_arr, int n_scales) {
    vector<Mat> pyr;
    int img_vec_size = pyr_i_arr.size();
    for (int scale = 0; scale < n_scales; scale++) {
        Mat cur_i = Mat::zeros(pyr_w_arr[0][scale].size(), CV_16SC1);
        for (int i = 0; i < cur_i.rows; i++) {
            for (int j = 0; j < cur_i.cols; j++) {
                float value = 0.0, weight = 0.0;
                for (int m = 0; m < img_vec_size; m++) {
                    //Ȩ�أ���Ȩ�ص��ۼ��ܺͣ����ڹ�һ��Ȩ��
                    weight += pyr_w_arr[m][scale].at<float>(i, j);
                    //��Ȩ��Ĳв��ܺ�
                    value += pyr_i_arr[m][scale].at<short>(i, j) * pyr_w_arr[m][scale].at<float>(i, j);
                }
                //����õ���Ȩƽ��ֵ
                value = value / weight;//��1��Ϊ�˷�ֹ��0
                cur_i.at<short>(i, j) = value;
            }
        }
        pyr.push_back(cur_i);
        //pyr_i_arr[0][scale].convertTo(pyr_i_arr[0][scale], CV_8UC1);
    }

    return pyr;
}

//�ع�
Mat fusion::PyrBuild(vector<Mat> pyr, int n_scales) {
    Mat out = pyr[n_scales - 1].clone();
    for (int i = n_scales - 2; i >= 0; i--) {
        resize(out, out, pyr[i].size());//�ϲ���
        for (int m = 0; m < out.rows; m++) {
            short* ptr_out = out.ptr<short>(m);
            short* ptr_i = pyr[i].ptr<short>(m);
            for (int n = 0; n < out.cols; n++) {
                ptr_out[n] = ptr_out[n] + ptr_i[n];
            }
        }
    }
    Mat out2;
    normalize(out, out2, 0, 255, NORM_MINMAX);
    out2.convertTo(out, CV_8UC1);

    return out;
}

//���������
Mat fusion::PyrResult(vector<Mat> src_arr, vector<Mat> weight_arr, int n_scales) {
    vector<vector<Mat>> pyr_i_arr, pyr_w_arr;
    for (int i = 0; i < src_arr.size(); i++) {
        vector<Mat> pyr_m1 = GaussianPyramid(weight_arr[i], n_scales);
        vector<Mat> pyr_s1 = LaplacianPyramid(src_arr[i], n_scales);
        Mat tmpt;
        Mat dst;
        //Ȩ�ؽ���������
        //for (int i = 0; i < pyr_m1.size() - 1; i++) {
        //    //���ٵ����˲���ע������ѭ����ֹ��������Ϊ���һ���ֱ���2X2���������²���4
        //    /*imwrite(".\\tmpt.bmp", pyr_m1[i]);
        //    tmpt = imread(".\\tmpt.bmp",0);
        //    dst = gb.guide(tmpt, 5, 0.0001, 4);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);*/
        //    //��˹�˲�
        //    //GaussianBlur(pyr_m1[i], pyr_m1[i], Size(5, 5), 0, 0);
        //    //��˹��
        //    /*GaussianBlur(pyr_m1[i], dst, Size(5, 5), 0, 0);
        //    addWeighted(pyr_m1[i], 1.5, dst, -0.5, 0, dst);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);*/
        //    //˫���˲�
        //    imwrite(".\\tmpt.bmp", pyr_m1[i]);
        //    tmpt = imread(".\\tmpt.bmp", 0);
        //    bilateralFilter(tmpt, dst, 5, 50, 50);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);
        //    //������˹��
        //    //sharpenImage(pyr_m1[i]);
        //    //�����ݶ���
        //    /*imwrite(".\\tmpt.bmp", pyr_m1[i]);
        //    tmpt = imread(".\\tmpt.bmp", 0);
        //    grad_sharp(tmpt, dst);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);*/
        //}
        //ԭʼͼ��������˹����������
        for (int i = 0; i < pyr_s1.size()- 1; i++) {
            //���ٵ����˲���ע������ѭ����ֹ��������Ϊ���һ���ֱ���2X2���������²���4
            imwrite(".\\tmpt.bmp", pyr_s1[i]);
            tmpt = imread(".\\tmpt.bmp",0);
            dst = gb.guide(tmpt, 5, 0.0001, 4);
            pyr_s1[i].convertTo(dst, CV_32FC3);
            //��˹�˲�
            //GaussianBlur(pyr_s1[i], pyr_s1[i], Size(5, 5), 0, 0);
            //��˹��
            /*GaussianBlur(pyr_s1[i], dst, Size(5, 5), 0, 0);
            addWeighted(pyr_s1[i], 1.5, dst, -0.5, 0, dst);
            pyr_s1[i].convertTo(dst, CV_32FC3);*/
            //˫���˲�
            /*imwrite(".\\tmpt.bmp", pyr_s1[i]);
            tmpt = imread(".\\tmpt.bmp", 0);
            bilateralFilter(tmpt, dst, 5, 50, 50);
            pyr_s1[i].convertTo(dst, CV_32FC3);*/
            //������˹��
            sharpenImage(pyr_s1[i]);
            //�����ݶ���
            /*imwrite(".\\tmpt.bmp", pyr_s1[i]);
            tmpt = imread(".\\tmpt.bmp", 0);
            grad_sharp(tmpt, dst);
            pyr_s1[i].convertTo(dst, CV_32FC3);*/
        }
        pyr_w_arr.push_back(pyr_m1);
        pyr_i_arr.push_back(pyr_s1);
    }
    vector<Mat> pyr = EdgeFusion(pyr_i_arr, pyr_w_arr, n_scales);
    Mat out = PyrBuild(pyr, n_scales);

    return out;
}

//�������ں�
Mat fusion::PyrFusion(vector<Mat> src_arr, vector<Mat> weight_arr) {
    int h = src_arr[0].rows;
    int w = src_arr[0].cols;
    int n_sc_ref = int(log(min(h, w)) / log(2));
    int n_scales = 1;
    while (n_scales < n_sc_ref) {
        n_scales++;
    }

    vector<Mat> channels_b, channels_g, channels_r;
    for (int i = 0; i < weight_arr.size(); i++) {
        vector<Mat> channels;
        split(src_arr[i], channels);
        channels_b.push_back(channels[0]);
        channels_g.push_back(channels[1]);
        channels_r.push_back(channels[2]);
    }

    Mat b = PyrResult(channels_b, weight_arr, n_scales);
    Mat g = PyrResult(channels_g, weight_arr, n_scales);
    Mat r = PyrResult(channels_r, weight_arr, n_scales);

    Mat out;
    vector<Mat> channels = { b, g, r };
    merge(channels, out);

    return out;
}

//����
Mat fusion::Run(vector<Mat> src_arr) {
    vector<Mat> weight_arr = WeightCalculate(src_arr);
    Mat out = PyrFusion(src_arr, weight_arr);

    return out;
}
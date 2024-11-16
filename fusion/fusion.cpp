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

// 锐化图像的函数
void sharpenImage(Mat image)
{
    //Mat sharpened;
    // 使用拉普拉斯滤波器进行图像锐化
    Mat kernel = (Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);
    filter2D(image, image, image.depth(), kernel);
    //return sharpened;
}

//基于梯度锐化
void grad_sharp(Mat image, Mat dst)
{
    // TODO: 在此添加控件通知处理程序代码
    Mat result, grad;

    result.create(image.size(), CV_8UC1);
    //grad.create(image.size(), CV_8UC1);

    //因为最后一行最后一列没有办法计算梯度所以，使用原图填充，先试用灰度图进行初始化
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



//计算图像亮度权重
Mat fusion::LightCalculate(Mat src)
{
	Mat light_weight = Mat::zeros(src.size(), CV_32FC1);
	float lut[256];
    //计算查找表，lut是一个数组，lut[i]是一个值，i是输入图像的像素值，lut[i]是输出图像的像素值
    //使用循环遍历0到255的每一个亮度值，并计算其对应的亮度权重
    for (int i = 0; i < 256; i++)
    {
        //将像素值归一化为范围[0, 1]，然后减去0.5，这个操作的目的可能是使得亮度值以0为中心
		float value = pow(i / 255.0 - 0.5, 2);
        //0.2作为高斯函数的方差，使得图像的亮度变化更加平滑
        //用来控制函数的陡峭程度，即控制权重变化的速率和范围
		lut[i] = exp(-value / (2 * (0.2 * 0.2)));
	}

    Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
            //lut:查找表,src_gray.at<uchar>(i, j):输入图像,light_weight.at<float>(i, j):输出图像
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
//高斯金字塔
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

//拉普拉斯金字塔
vector<Mat> fusion::LaplacianPyramid(Mat img, int level) {
    vector<Mat> pyr;
    Mat item = img;
    for (int i = 1; i < level; i++) {
        Mat item_down;
        Mat item_up;
        //下采样，得到低分辨率图像
        resize(item, item_down, item.size() / 2, 0, 0, INTER_AREA);
        //上采样到原始 item 的尺寸，并存储在 item_up 中
        resize(item_down, item_up, item.size());

        //拉普拉斯金字塔的每一层的残差
        Mat diff(item.size(), CV_16SC1);
        for (int m = 0; m < item.rows; m++) {
            short* ptr_diff = diff.ptr<short>(m);
            uchar* ptr_up = item_up.ptr(m);
            uchar* ptr_item = item.ptr(m);

            for (int n = 0; n < item.cols; n++) {
                ptr_diff[n] = (short)ptr_item[n] - (short)ptr_up[n];//求残差
            }
        }
        pyr.push_back(diff);
        item = item_down;
        //imwrite(format(".\\laplacian_%d.bmp", i), diff);
    }
    //添加原始图像，到金字塔的末尾
    item.convertTo(item, CV_16SC1);
    pyr.push_back(item);

    return pyr;
}

//边缘融合
//将拉普拉斯金字塔的每一层的残差加权求和，得到新的残差，然后重构
//pyr_i_arr:要融合图像的拉普拉斯金字塔,pyr_w_arr:权重金字塔、高斯金字塔
vector<Mat> fusion::EdgeFusion(vector<vector<Mat>> pyr_i_arr, vector<vector<Mat>> pyr_w_arr, int n_scales) {
    vector<Mat> pyr;
    int img_vec_size = pyr_i_arr.size();
    for (int scale = 0; scale < n_scales; scale++) {
        Mat cur_i = Mat::zeros(pyr_w_arr[0][scale].size(), CV_16SC1);
        for (int i = 0; i < cur_i.rows; i++) {
            for (int j = 0; j < cur_i.cols; j++) {
                float value = 0.0, weight = 0.0;
                for (int m = 0; m < img_vec_size; m++) {
                    //权重，是权重的累加总和，用于归一化权重
                    weight += pyr_w_arr[m][scale].at<float>(i, j);
                    //加权后的残差总和
                    value += pyr_i_arr[m][scale].at<short>(i, j) * pyr_w_arr[m][scale].at<float>(i, j);
                }
                //计算得到加权平均值
                value = value / weight;//加1是为了防止除0
                cur_i.at<short>(i, j) = value;
            }
        }
        pyr.push_back(cur_i);
        //pyr_i_arr[0][scale].convertTo(pyr_i_arr[0][scale], CV_8UC1);
    }

    return pyr;
}

//重构
Mat fusion::PyrBuild(vector<Mat> pyr, int n_scales) {
    Mat out = pyr[n_scales - 1].clone();
    for (int i = n_scales - 2; i >= 0; i--) {
        resize(out, out, pyr[i].size());//上采样
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

//金字塔结果
Mat fusion::PyrResult(vector<Mat> src_arr, vector<Mat> weight_arr, int n_scales) {
    vector<vector<Mat>> pyr_i_arr, pyr_w_arr;
    for (int i = 0; i < src_arr.size(); i++) {
        vector<Mat> pyr_m1 = GaussianPyramid(weight_arr[i], n_scales);
        vector<Mat> pyr_s1 = LaplacianPyramid(src_arr[i], n_scales);
        Mat tmpt;
        Mat dst;
        //权重金字塔处理
        //for (int i = 0; i < pyr_m1.size() - 1; i++) {
        //    //快速导向滤波，注意上面循环终止条件，因为最后一个分辨率2X2，不满足下采样4
        //    /*imwrite(".\\tmpt.bmp", pyr_m1[i]);
        //    tmpt = imread(".\\tmpt.bmp",0);
        //    dst = gb.guide(tmpt, 5, 0.0001, 4);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);*/
        //    //高斯滤波
        //    //GaussianBlur(pyr_m1[i], pyr_m1[i], Size(5, 5), 0, 0);
        //    //高斯锐化
        //    /*GaussianBlur(pyr_m1[i], dst, Size(5, 5), 0, 0);
        //    addWeighted(pyr_m1[i], 1.5, dst, -0.5, 0, dst);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);*/
        //    //双边滤波
        //    imwrite(".\\tmpt.bmp", pyr_m1[i]);
        //    tmpt = imread(".\\tmpt.bmp", 0);
        //    bilateralFilter(tmpt, dst, 5, 50, 50);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);
        //    //拉普拉斯锐化
        //    //sharpenImage(pyr_m1[i]);
        //    //基于梯度锐化
        //    /*imwrite(".\\tmpt.bmp", pyr_m1[i]);
        //    tmpt = imread(".\\tmpt.bmp", 0);
        //    grad_sharp(tmpt, dst);
        //    pyr_m1[i].convertTo(dst, CV_32FC3);*/
        //}
        //原始图像拉普拉斯金字塔处理
        for (int i = 0; i < pyr_s1.size()- 1; i++) {
            //快速导向滤波，注意上面循环终止条件，因为最后一个分辨率2X2，不满足下采样4
            imwrite(".\\tmpt.bmp", pyr_s1[i]);
            tmpt = imread(".\\tmpt.bmp",0);
            dst = gb.guide(tmpt, 5, 0.0001, 4);
            pyr_s1[i].convertTo(dst, CV_32FC3);
            //高斯滤波
            //GaussianBlur(pyr_s1[i], pyr_s1[i], Size(5, 5), 0, 0);
            //高斯锐化
            /*GaussianBlur(pyr_s1[i], dst, Size(5, 5), 0, 0);
            addWeighted(pyr_s1[i], 1.5, dst, -0.5, 0, dst);
            pyr_s1[i].convertTo(dst, CV_32FC3);*/
            //双边滤波
            /*imwrite(".\\tmpt.bmp", pyr_s1[i]);
            tmpt = imread(".\\tmpt.bmp", 0);
            bilateralFilter(tmpt, dst, 5, 50, 50);
            pyr_s1[i].convertTo(dst, CV_32FC3);*/
            //拉普拉斯锐化
            sharpenImage(pyr_s1[i]);
            //基于梯度锐化
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

//金字塔融合
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

//运行
Mat fusion::Run(vector<Mat> src_arr) {
    vector<Mat> weight_arr = WeightCalculate(src_arr);
    Mat out = PyrFusion(src_arr, weight_arr);

    return out;
}
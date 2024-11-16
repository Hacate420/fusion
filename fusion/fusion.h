#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "guideblur.h"

using namespace cv;
using namespace std;

class fusion
{
public:
    fusion();
	~fusion();
	Mat Run(vector<Mat> src_arr);

private:
    Mat ContrastCalculate(Mat src);
    Mat SaturationCalculate(Mat src);
    Mat LightCalculate(Mat src);
    vector<Mat> WeightCalculate(vector<Mat> src_arr);

    vector<Mat> GaussianPyramid(Mat img, int level);
    vector<Mat> LaplacianPyramid(Mat img, int level);
    Mat PyrBuild(vector<Mat> pyr, int n_scales);
    vector<Mat> EdgeFusion(vector<vector<Mat>> pyr_i_arr, vector<vector<Mat>> pyr_w_arr, int n_scales);

    Mat PyrResult(vector<Mat> src_arr, vector<Mat> weight_arr, int n_scales);
    Mat PyrFusion(vector<Mat> src_arr, vector<Mat> weight_arr);

};

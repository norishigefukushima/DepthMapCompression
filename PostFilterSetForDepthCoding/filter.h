#ifndef _FILTER_H_
#define _FILTER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//rgb interleave function for bilateral filter
void splitBGRLineInterleave( const Mat& src, Mat& dest);

void smallGaussianBlur(const Mat& src, Mat& dest, const int d, const double sigma);

//max, min filter and blur remove filter by using min-max filter
void maxFilter(const Mat& src, Mat& dest, Size ksize, int borderType=cv::BORDER_REPLICATE);
void minFilter(const Mat& src, Mat& dest, Size ksize, int borderType=cv::BORDER_REPLICATE);
void blurRemoveMinMax(Mat& src, Mat& dest, const int r);
void blurRemoveMinMax(Mat& src, Mat& dest, const int r, const int threshold);
void blurRemoveMinMaxBF(Mat& src, Mat& dest, const int r, const int threshold);

//range filter functions
enum
{
	FULL_KERNEL = 0,
	SEPARABLE_KERNEL
};
void binalyWeightedRangeFilter(const Mat& src, Mat& dst, Size kernelSize, float threshold, int method, int borderType=cv::BORDER_REPLICATE);

//post filter set class
class PostFilterSet
{
	Mat buff,bufff;
public:
	PostFilterSet();
	~PostFilterSet();
	void filterDisp8U2Depth32F(Mat& src, Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method=FULL_KERNEL);
	void filterDisp8U2Depth16U(Mat& src, Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method=FULL_KERNEL);
	void filterDisp8U2Disp32F(Mat& src, Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, float brange_th, int brange_method=FULL_KERNEL);
	void operator()(Mat& src, Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, int brange_th, int brange_method=FULL_KERNEL);
};

//boundary reconstruction filter for lossy encoded depth maps
void boundaryReconstructionFilter(Mat& src, Mat& dest, Size ksize, const float frec, const float color, const float space);

#endif
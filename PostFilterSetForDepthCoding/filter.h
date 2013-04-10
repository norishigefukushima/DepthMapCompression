#define HAVE_TBB
#ifndef _FILTER_H_
#define _FILTER_H_

#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <smmintrin.h>

#pragma comment(lib, "tbb.lib")
#ifdef _DEBUG
//#pragma comment(lib, "opencv_video240d.lib")
//#pragma comment(lib, "opencv_ts240d.lib")
//#pragma comment(lib, "opencv_stitching240d.lib")
//#pragma comment(lib, "opencv_photo240d.lib")
//#pragma comment(lib, "opencv_objdetect240d.lib")
//#pragma comment(lib, "opencv_ml240d.lib")
//#pragma comment(lib, "opencv_legacy240d.lib")
#pragma comment(lib, "opencv_imgproc240d.lib")
#pragma comment(lib, "opencv_highgui240d.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
//#pragma comment(lib, "opencv_gpu240d.lib")
//#pragma comment(lib, "opencv_flann240d.lib")
//#pragma comment(lib, "opencv_features2d240d.lib")
#pragma comment(lib, "opencv_core240d.lib")
#pragma comment(lib, "opencv_contrib240d.lib")
#pragma comment(lib, "opencv_calib3d240d.lib")
#else
//#pragma comment(lib, "opencv_video240.lib")
//#pragma comment(lib, "opencv_ts240.lib")
//#pragma comment(lib, "opencv_stitching240.lib")
//#pragma comment(lib, "opencv_photo240.lib")
//#pragma comment(lib, "opencv_objdetect240.lib")
//#pragma comment(lib, "opencv_ml240.lib")
//#pragma comment(lib, "opencv_legacy240.lib")
#pragma comment(lib, "opencv_imgproc240.lib")
//#pragma comment(lib, "opencv_highgui240.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
//#pragma comment(lib, "opencv_gpu240.lib")
//#pragma comment(lib, "opencv_flann240.lib")
//#pragma comment(lib, "opencv_features2d240.lib")
#pragma comment(lib, "opencv_core240.lib")
//#pragma comment(lib, "opencv_contrib240.lib")
//#pragma comment(lib, "opencv_calib3d240.lib")
#endif
using namespace cv;
using namespace std;

#define CV_SSE4_1 1

void depth16U2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a=1.f, float b=0.f);
void disp8U2depth32F(Mat& src, Mat& dest, const float focal_baseline, float a=1.f, float b=0.f);

void smallGaussianBlur(const Mat& src, Mat& dest, const int d, const double sigma);
//boundary reconstruction filter for lossy encoded depth maps
void boundaryReconstructionFilter(Mat& src, Mat& dest, Size ksize, const float frec, const float color, const float space);

//max, min filter and blur remove filter by using min-max filter
void maxFilter(const Mat& src, Mat& dest, Size ksize, int borderType=cv::BORDER_REPLICATE);
void minFilter(const Mat& src, Mat& dest, Size ksize, int borderType=cv::BORDER_REPLICATE);
void blurRemoveMinMax(Mat& src, Mat& dest, const int r);
void blurRemoveMinMax(Mat& src, Mat& dest, const int r, const int threshold);
void blurRemoveMinMaxBF(Mat& src, Mat& dest, const int r, const int threshold);



//rgb interleave function for bilateral filter
void splitBGRLineInterleave( const Mat& src, Mat& dest);

//bilateral filter functions
enum
{
	BILATERAL_NORMAL = 0,
	BILATERAL_SEPARABLE,
	BILATERAL_ORDER2,//underconstruction
	BILATERAL_ORDER2_SEPARABLE//underconstruction
};

void binalyWeightedRangeFilterSingle_32f( const Mat& src, Mat& dst, Size kernelSize, float threshold, int borderType=cv::BORDER_REPLICATE);
void binalyWeightedRangeFilter(const Mat& src, Mat& dst, Size kernelSize, float threshold, int method, int borderType=cv::BORDER_REPLICATE);
void bilateralFilterBase( const Mat& src, Mat& dst, int d,
	double sigma_color, double sigma_space,int borderType=cv::BORDER_REPLICATE);
void bilateralWeightMapBase( const Mat& src, Mat& dst, int d,
	double sigma_color, double sigma_space,int borderType=cv::BORDER_REPLICATE);

void bilateralFilter(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);
void weightedBilateralFilter(const Mat& src, Mat& weight, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);
void bilateralWeightMap(const Mat& src, Mat& dst, Size kernelSize, double sigma_color, double sigma_space, int method=BILATERAL_NORMAL, int borderType=cv::BORDER_REPLICATE);

class PostFilterSet
{
	Mat temp,tempf;
public:
	PostFilterSet(){;}
	~PostFilterSet(){;}

	void filterDisp8U2Depth16U(Mat& src, Mat& dest, double focus, double baseline, double amp, int median_r, int gaussian_r, int minmax_r, int brange_r, int brange_th, int brange_method=BILATERAL_NORMAL)
	{
		medianBlur(src,temp,2*median_r+1);	
		smallGaussianBlur(temp,temp,2*gaussian_r+1,gaussian_r+0.5);
		blurRemoveMinMax(temp,temp,minmax_r);

		disp8U2depth32F(temp,tempf,focus*baseline,amp,0.f);

		binalyWeightedRangeFilter(tempf,tempf,Size(2*brange_r+1,2*brange_r+1),brange_th,brange_method);

		tempf.convertTo(dest,CV_16U);
	}

	void operator()(Mat& src, Mat& dest, int median_r, int gaussian_r, int minmax_r, int brange_r, int brange_th, int brange_method=BILATERAL_NORMAL)
	{
		medianBlur(src,temp,2*median_r+1);	
		smallGaussianBlur(temp,temp,2*gaussian_r+1,gaussian_r+0.5);
		blurRemoveMinMax(temp,temp,minmax_r);
		binalyWeightedRangeFilter(temp,dest,Size(2*brange_r+1,2*brange_r+1),brange_th,brange_method);
	}
};
#endif
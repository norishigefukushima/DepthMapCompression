#ifndef _UTIL_H_
#define _UTIL_H_

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//point cloud rendering
Point3d get3DPointfromXYZ(Mat& xyz, Size& imsize, Point& pt);
void reprojectXYZ(const Mat& depth, Mat& xyz, Mat& intrinsic, Mat& distortion, float a=1.0, float b=0.0);
void reprojectXYZ(const Mat& depth, Mat& xyz, double f);
void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub);
void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub, vector<Point2f>& pt, Mat& depth);

//oocclusion filling
enum
{
	FILL_DISPARITY =0,
	FILL_DEPTH =1
};
void fillOcclusion(Mat& src, int invalidvalue, int disp_or_depth=FILL_DEPTH);
void fillSmallHole(const Mat& src, Mat& dest);

//disparity depth converter
void depth32F2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a=1.f, float b=0.f);
void disp16S2depth16U(Mat& src, Mat& dest, const float focal_baseline, float a=1.f, float b=0.f);
void depth16U2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a=1.f, float b=0.f);
void disp8U2depth32F(Mat& src, Mat& dest, const float focal_baseline, float a=1.f, float b=0.f);

void projectPointsSimple(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest);//multi points projection
void projectPointSimple(Point3d& xyz, const Mat& R, const Mat& t, const Mat& K, Point2d& dest);//single point projection
//geometric functions
void lookat(const Point3d& from, const Point3d& to, Mat& destR);
void eular2rot(double pitch, double roll, double yaw, Mat& dest);
void setXYZ(Mat& in, double&x, double&y, double&z);
void point3d2Mat(const Point3d& src, Mat& dest);
void rotPitch(Mat& src, Mat& dest, const double pitch);
void rotYaw(Mat& src, Mat& dest, const double yaw);

//coding functions
void degradeJPEG(const Mat& src, Mat& dest, int q, int DCT_MODE, bool isOpt, int& size, double& bpp);
double degradeImagex264(Mat& src, Mat& dest, int qp, int& size, double& bpp);

//test functions
void guiAlphaBlend(Mat& src1, Mat& src2);
double getPSNR(Mat& src1, Mat& src2);
double getPSNR(Mat& src1, Mat& src2, int bb);

void showDiffPoint(Mat& src1, Mat& src2);

class ConsoleImage
{
private:
	int count;
	std::vector<std::string> strings;

public:
	cv::Mat show;
	ConsoleImage(cv::Size size=Size(640,480));
	~ConsoleImage();
	void printData();
	void clear();
	void operator()(string src);
	void operator()(const char *format, ...);
	void operator()(cv::Scalar color, const char *format, ...);
};

enum
{
	TIME_NSEC=0,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR
};

class CalcTime
{
	int64 pre;
	string mes;

	int timeMode;

	double cTime;
	bool _isShow;


public:

	void start();
	void setMode(int mode);//単位
	void setMessage(string src);
	void restart();//再計測開始
	double getTime();//時間を取得
	void show();//cout<< time
	void show(string message);//cout<< time

	CalcTime(string message="time ", int mode=TIME_MSEC ,bool isShow=true);
	~CalcTime();
};

#endif
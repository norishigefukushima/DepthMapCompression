#ifndef _UTIL_H_
#define _UTIL_H_
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void degradeJPEG(const Mat& src, Mat& dest, int q, int DCT_MODE, bool isOpt, int& size, double& bpp);
double degradeImagex264(Mat& src, Mat& dest, int qp, int& size, double& bpp);

void projectTest();


void guiAlphaBlend(Mat& src1, Mat& src2);
double getPSNR(Mat& src1, Mat& src2);

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

#define XCV_WHITE CV_RGB(255,255,255)
#define XCV_GRAY10 CV_RGB(10,10,10)
#define XCV_GRAY20 CV_RGB(20,20,20)
#define XCV_GRAY30 CV_RGB(10,30,30)
#define XCV_GRAY40 CV_RGB(40,40,40)
#define XCV_GRAY50 CV_RGB(50,50,50)
#define XCV_GRAY60 CV_RGB(60,60,60)
#define XCV_GRAY70 CV_RGB(70,70,70)
#define XCV_GRAY80 CV_RGB(80,80,80)
#define XCV_GRAY90 CV_RGB(90,90,90)
#define XCV_GRAY100 CV_RGB(100,100,100)
#define XCV_GRAY110 CV_RGB(101,110,110)
#define XCV_GRAY120 CV_RGB(120,120,120)
#define XCV_GRAY130 CV_RGB(130,130,140)
#define XCV_GRAY140 CV_RGB(140,140,140)
#define XCV_GRAY150 CV_RGB(150,150,150)
#define XCV_GRAY160 CV_RGB(160,160,160)
#define XCV_GRAY170 CV_RGB(170,170,170)
#define XCV_GRAY180 CV_RGB(180,180,180)
#define XCV_GRAY190 CV_RGB(190,190,190)
#define XCV_GRAY200 CV_RGB(200,200,200)
#define XCV_GRAY210 CV_RGB(210,210,210)
#define XCV_GRAY220 CV_RGB(220,220,220)
#define XCV_GRAY230 CV_RGB(230,230,230)
#define XCV_GRAY240 CV_RGB(240,240,240)
#define XCV_GRAY250 CV_RGB(250,250,250)
#define XCV_BLACK CV_RGB(0,0,0)

#define XCV_RED CV_RGB(255,0,0)
#define XCV_GREEN CV_RGB(0,255,0)
#define XCV_BLUE CV_RGB(0,0,255)
#define XCV_ORANGE CV_RGB(255,100,0)
#define XCV_YELLOW CV_RGB(255,255,0)
#define XCV_MAGENDA CV_RGB(255,0,255)
#define XCV_CYAN CV_RGB(0,255,255)

#define XCV_KEY_NAME_MAX 32
#define XCV_PLOTDATA_MAX 64

enum
{
	XCV_PLOT_NOPOINT = 0,
	XCV_PLOT_PLUS,
	XCV_PLOT_CROSS,
	XCV_PLOT_ASTERRISK,
	XCV_PLOT_CIRCLE,
	XCV_PLOT_RECTANGLE,
	XCV_PLOT_CIRCLE_FILL,
	XCV_PLOT_RECTANGLE_FILL,
	XCV_PLOT_TRIANGLE,
	XCV_PLOT_TRIANGLE_FILL,
	XCV_PLOT_TRIANGLE_INV,
	XCV_PLOT_TRIANGLE_INV_FILL,
};

enum
{
	XCV_LINE_NONE,
	XCV_LINE_LINEAR,
	XCV_LINE_H2V,
	XCV_LINE_V2H
};

enum
{
	XCV_DRAWCROSS_PLUS = 0,
	XCV_DRAWCROSS_TIMES,
	XCV_DRAWCROSS_ASTERRISK
};

void cvtPseudoColor(cv::Mat& gray, cv::Mat& color);
void cvtPseudoColor2(cv::Mat& gray, cv::Mat& color);
void triangle(cv::Mat& src,cv::Point pt,int length, cv::Scalar color, int thickness=1);
void triangleinv(cv::Mat& src,cv::Point pt,int length, cv::Scalar color, int thickness=1);
void drawCross(cv::Mat& dest,cv::Point crossCenter, int length, cv::Scalar color, int mode=XCV_DRAWCROSS_PLUS,int thickness=1, int line_type=8,int shift=0);
void drawGrid(cv::Mat& dest, cv::Point point, cv::Scalar color, int thickness=1, int line_type=8, int shift=0);

void xcvTriangle(IplImage* src, CvPoint pt, int length, CvScalar color, int thickness=1);
void xcvTriangleInv(IplImage* src, CvPoint pt, int length, CvScalar color, int thickness=1);
void xcvDrawGrid(IplImage* src,CvPoint gridCenter,CvScalar color,int thickness=1, int line_type=8,int shift=0);
void xcvDrawCross(IplImage* dest,CvPoint crossCenter, int length,CvScalar color, int mode=XCV_DRAWCROSS_PLUS,int thickness=1, int line_type=8,int shift  = 0);
void xcvRectangle(IplImage* dest,CvRect rect, CvScalar color,int thickness=1, int line_type=8,int shift=0);

void xcvPutText(IplImage* render,char* text,CvPoint orign, CvScalar color=CV_RGB(255,255,255), double amp=1.0, double shear=0.0, int fontType=CV_FONT_HERSHEY_SIMPLEX, int thickcness=1);


void xcvPlotData(IplImage* render,CvMat* data,int data_size,double xmin,double xmax, double ymin, double ymax,
	CvScalar color=XCV_RED, int lt=XCV_PLOT_PLUS, int isLine=XCV_LINE_LINEAR,int thickness=1, int ps=4);

void xcvPlotData(IplImage* render,CvPoint2D32f* data,int data_size,double xmin,double xmax, double ymin, double ymax,
	CvScalar color=XCV_RED, int lt=XCV_PLOT_PLUS, bool isLine=XCV_LINE_LINEAR,int thickness=1, int ps=4);


class CV_EXPORTS Plot
{
protected:
	char xlabel[256];
	char ylabel[256];
	char keyname[XCV_PLOTDATA_MAX][XCV_KEY_NAME_MAX];

	int data_max;
	int point_max;
	
	CvMat* data[XCV_PLOTDATA_MAX];
	int matsize[XCV_PLOTDATA_MAX];
	int data_size[XCV_PLOTDATA_MAX];
	CvScalar color[XCV_PLOTDATA_MAX];
	int lt[XCV_PLOTDATA_MAX];
	int thickness[XCV_PLOTDATA_MAX];
	int isLine[XCV_PLOTDATA_MAX];

	CvScalar background_color;

	CvSize plotsize;
	CvPoint origin;

	double xmin;
	double xmax;
	double ymin;
	double ymax;
	double xmax_no_margin;
	double xmin_no_margin;
	double ymax_no_margin;
	double ymin_no_margin;

	void Plot::init();

	void Plot::free();
	void Plot::point2val(CvPoint pt, double* valx, double* valy);

	bool isZeroCross;
	bool isXYMAXMIN;
	bool isXYCenter;

	bool isPosition;
	cv::Scalar getPseudoColor(uchar val);
	IplImage* plotImage;
	IplImage* render;
	IplImage* keyImage;
public:
	cv::Mat renderMat;
	cv::Mat graphImage;

	Plot::Plot(cv::Size window_size = cv::Size(1024,768));
	Plot::~Plot();

	void Plot::setXYOriginZERO();
	void Plot::setXOriginZERO();
	void Plot::setYOriginZERO();

	void Plot::recomputeXYMAXMIN(bool isCenter = false, double marginrate = 0.9);
	void Plot::setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_,bool isZeroCross_);
	void Plot::setPlotImageSize(cv::Size s);
	void Plot::setXYMinMax(double xmin_,double xmax_,double ymin_,double ymax_);
	void Plot::setXMinMax(double xmin_,double xmax_);
	void Plot::setYMinMax(double ymin_,double ymax_);
	void Plot::setBackGoundColor(cv::Scalar cl);

	void Plot::add(std::vector<cv::Point> point, int num=0);
	void Plot::add(std::vector<cv::Point2d> point, int num=0);
	void Plot::add(std::vector<cv::Point2f> point, int num=0);
	void Plot::add(double x, double y, int num=0);
	void Plot::clear(int datanum=-1);

	void Plot::makeBB(bool isFont);
	void Plot::setPlot(int plotnum,CvScalar color_=XCV_RED,int lt_=XCV_PLOT_PLUS, int isLine_=XCV_LINE_LINEAR,int thickness_ = 1);
	void Plot::setLinetypeALL(int linetype);

	void Plot::plotPoint(cv::Point2d = cv::Point2d(0.0,0.0) , CvScalar color_ = XCV_BLACK, int thickness_=1,int linetype=XCV_LINE_LINEAR);
	void Plot::plotGrid(int level);
	void Plot::plotData(int gridlevel=0,int isKey=0);

	void Plot::plot(char* name="Plot");
	void Plot::makeKey(int num);
	void Plot::setKeyName(char* name, int num);

	void Plot::save(char* name);
};
#endif
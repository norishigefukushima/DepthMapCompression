#include "util.h"

double getPSNR(Mat& src1, Mat& src2)
{
	CV_Assert(src1.channels()==src2.channels() && src1.type()==src2.type() && src1.data !=src2.data);
	double psnr;

	Mat s1,s2;
	if(src1.channels()==3)
	{
		cvtColor(src1,s1,CV_BGR2GRAY);
		cvtColor(src2,s2,CV_BGR2GRAY);
	}
	else
	{
		s1=src1;
		s2=src2;
	}

	//cout<<s1.cols<<","<<s1.rows<<endl;
	//cout<<s2.cols<<","<<s2.rows<<endl;
	Mat sub;
	subtract(s1,s2,sub,Mat(),CV_32F);
	multiply(sub,sub,sub);

	int count = s1.size().area();
	Scalar v = cv::mean(sub);

	if(v.val[0] == 0.0 || count==0)
	{
		return -1;
	}
	else
	{
		psnr = 10.0*log10((255.0*255.0)/v.val[0]);
		return psnr;
	}
}


void CalcTime::start()
{
	pre = getTickCount();
}

void CalcTime::restart()
{
	start();
}

void CalcTime:: show()
{
	getTime();
	switch(timeMode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;

	case TIME_MSEC:
	default:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	}
}

void CalcTime:: show(string mes)
{
	getTime();
	switch(timeMode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;

	case TIME_MSEC:
	default:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	}
}

double CalcTime:: getTime()
{
	cTime = (getTickCount()-pre)/(getTickFrequency());
	switch(timeMode)
	{
	case TIME_NSEC:
		cTime*=1000000.0;
		break;
	case TIME_SEC:
		cTime*=1.0;
		break;
	case TIME_MIN:
		cTime /=(60.0);
		break;
	case TIME_HOUR:
		cTime /=(60*60);
		break;
	case TIME_MSEC:
	default:
		cTime *=1000.0;
		break;
	}
	return cTime;
}
void CalcTime:: setMessage(string src)
{
	mes=src;
}
void CalcTime:: setMode(int mode)
{
	timeMode = mode;
}
CalcTime::CalcTime(string message,int mode,bool isShow)
{
	_isShow = isShow;
	timeMode = mode;

	setMessage(message);
	start();
}
CalcTime::~CalcTime()
{
	getTime();
	if(_isShow)	show();
}

ConsoleImage::ConsoleImage(Size size)
{
	show = Mat::zeros(size, CV_8UC3);
	clear();
}
ConsoleImage::~ConsoleImage()
{
	printData();
}
void ConsoleImage::printData()
{
	for(int i=0;i<(int)strings.size();i++)
	{
		cout<<strings[i]<<endl;
	}
}
void ConsoleImage::clear()
{
	count = 0;
	show.setTo(0);
	strings.clear();
}

void ConsoleImage::operator()(string src)
{
	
	//CvFont font = fontQt("Times",16,CV_RGB(255,255,255));
	strings.push_back(src);
	//xcvPutText(&IplImage(show),(char*)src.c_str(),Point(20,20+count*20),CV_RGB(255,255,255),1,0,CV_FONT_HERSHEY_COMPLEX_SMALL);
	//addText(show,buff,Point(20,20+count*20),font);
	cv::putText(show,src,Point(20,20+count*20),CV_FONT_HERSHEY_COMPLEX_SMALL,1.0,CV_RGB(255,255,255),1);
	count++;
}
void ConsoleImage::operator()(const char *format, ...)
{
	char buff[255]; 

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;
	//CvFont font = fontQt("Times",16,CV_RGB(255,255,255));
	strings.push_back(a);
	//xcvPutText(&IplImage(show),buff,Point(20,20+count*20),CV_RGB(255,255,255),1,0,CV_FONT_HERSHEY_COMPLEX_SMALL);
	cv::putText(show,buff,Point(20,20+count*20),CV_FONT_HERSHEY_COMPLEX_SMALL,1.0,CV_RGB(255,255,255),1);
	//addText(show,buff,Point(20,20+count*20),font);
	count++;
}

void ConsoleImage::operator()(cv::Scalar color, const char *format, ...)
{
	char buff[255]; 

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;
	//CvFont font = fontQt("Times",16,CV_RGB(255,255,255));
	strings.push_back(a);
	//xcvPutText(&IplImage(show),buff,Point(20,20+count*20),color,1,0,CV_FONT_HERSHEY_COMPLEX_SMALL);
	cv::putText(show,buff,Point(20,20+count*20),CV_FONT_HERSHEY_COMPLEX_SMALL,1.0,CV_RGB(255,255,255),1);
	//addText(show,buff,Point(20,20+count*20),font);
	count++;
}

void guiAlphaBlend(Mat& src1, Mat& src2)
{
	Mat s1,s2;
	if(src1.channels()==1)cvtColor(src1,s1,CV_GRAY2BGR);
	else s1 = src1;
	if(src2.channels()==1)cvtColor(src2,s2,CV_GRAY2BGR);
	else s2 = src2;
	namedWindow("alphaBlend");
	int a = 0;
	createTrackbar("a","alphaBlend",&a,100);
	int key = 0;
	Mat show;
	while(key!='q')
	{
		addWeighted(s1,a/100.0,s2,1.0-a/100.0,0.0,show);
		imshow("alphaBlend",show);
		key = waitKey(1);
	}
	
}


void cvtPseudoColor2(Mat& gray, Mat& color)
{
	vector<Mat> c(3);
	c[0].create(gray.size(),CV_8U);
	c[1].create(gray.size(),CV_8U);
	c[2].create(gray.size(),CV_8U);
	
	double d = 255.0/63.0;

	{//g
		//int vmax = 200;
		int vmax = 255;
		double dv = (double)vmax/63.0;
		uchar lr[256];
		for (int i=0; i<64; i++)
			lr[i]=cvRound(d*i);
		for (int i=64; i<192; i++)
			lr[i]=vmax;
		for (int i=192; i<256; i++)
			lr[i]=cvRound(vmax-dv*(i-192));

		Mat lut(1, 256, CV_8UC1, lr);
		LUT(gray,c[1],lut);
	}
	{//r
		uchar lr[256];
		for (int i=0; i<128; i++)
			lr[i]=0;
		for (int i=128; i<192; i++)
			lr[i]=cvRound(d*(i-128));
		for (int i=192; i<256; i++)
			lr[i]=255;

		Mat lut(1, 256, CV_8UC1, lr);
		LUT(gray,c[2],lut);
	}
	{//b
		uchar lr[256];
		for (int i=0; i<64; i++)
			lr[i]=255;
		for (int i=64; i<128; i++)
			lr[i]=cvRound(255-d*(i-64));
		for (int i=128; i<256; i++)
			lr[i]=0;
		Mat lut(1, 256, CV_8UC1, lr);
		LUT(gray,c[0],lut);
	}
	merge(c,color);
}
void triangle(Mat& src,Point pt,int length, Scalar color, int thickness)
{
	int npt[] = {3, 0};
	cv::Point pt1[1][3];
	const int h = cvRound(1.7320508*0.5*length);
	pt1[0][0] = Point(pt.x,pt.y-h/2); ;
	pt1[0][1] = Point(pt.x+length/2,pt.y+h/2);
	pt1[0][2] = Point(pt.x-length/2,pt.y+h/2);

	const cv::Point *ppt[1] = {pt1[0]};

	if(thickness==CV_FILLED)
	{
		fillPoly(src, ppt, npt, 1, color,1);
	}
	else
	{
		polylines(src,ppt,npt,1,true,color,thickness);
	}
}

void triangleinv(Mat& src,Point pt,int length, Scalar color, int thickness)
{
	int npt[] = {3, 0};
	cv::Point pt1[1][3];
	const int h = cvRound(1.7320508*0.5*length);
	pt1[0][0] = Point(pt.x,pt.y+h/2); ;
	pt1[0][1] = Point(pt.x+length/2,pt.y-h/2);
	pt1[0][2] = Point(pt.x-length/2,pt.y-h/2);

	const cv::Point *ppt[1] = {pt1[0]};

	if(thickness==CV_FILLED)
	{
		fillPoly(src, ppt, npt, 1, color,1);
	}
	else
	{
		polylines(src,ppt,npt,1,true,color,thickness);
	}
}

void drawGrid(Mat& dest, Point point, Scalar color, int thickness, int line_type, int shift)
{
	xcvDrawGrid(&IplImage(dest),point,color,thickness,line_type,shift);
}

void drawCross(Mat& dest,Point crossCenter, int length, Scalar color, int mode,int thickness, int line_type,int shift)
{
	xcvDrawCross(&IplImage(dest),crossCenter,length,color,mode,thickness,line_type,shift);
}


void xcvTriangle(IplImage* src, CvPoint pt, int length, CvScalar color, int thickness)
{
	triangle(Mat(src),Point(pt),length,color,thickness);
}

void xcvTriangleInv(IplImage* src, CvPoint pt, int length, CvScalar color, int thickness)
{
	triangleinv(Mat(src),Point(pt),length,color,thickness);
}
void xcvDrawGrid(IplImage* src,CvPoint gridCenter,CvScalar color,int thickness, int line_type,int shift)
{
	CvPoint pt=gridCenter;
	if(pt.x==0 &&pt.y==0)
	{
		pt.x=src->width/2;
		pt.y=src->height/2;
	}
	cvLine(src,cvPoint(0,pt.y),cvPoint(src->width,pt.y),color, thickness, line_type, shift);
	cvLine(src,cvPoint(pt.x,0),cvPoint(pt.x,src->height),color, thickness, line_type, shift);
}

void xcvDrawCross(IplImage* dest,CvPoint crossCenter, int length, CvScalar color, int mode,int thickness, int line_type,int shift)
{
	if(crossCenter.x==0 &&crossCenter.y==0)
	{
		crossCenter.x=dest->width/2;
		crossCenter.y=dest->height/2;
	}
	if(mode == XCV_DRAWCROSS_PLUS)
	{
		int hl=length/2;
		cvLine(dest,cvPoint(crossCenter.x-hl,crossCenter.y),cvPoint(crossCenter.x+hl,crossCenter.y),color, thickness, line_type, shift);
		cvLine(dest,cvPoint(crossCenter.x,crossCenter.y-hl),cvPoint(crossCenter.x,crossCenter.y+hl),color, thickness, line_type, shift);
	}
	else if(mode == XCV_DRAWCROSS_TIMES)
	{
		int hl=cvRound((double)length/2.0/sqrt(2.0));
		cvLine(dest,cvPoint(crossCenter.x-hl,crossCenter.y-hl),cvPoint(crossCenter.x+hl,crossCenter.y+hl),color, thickness, line_type, shift);
		cvLine(dest,cvPoint(crossCenter.x+hl,crossCenter.y-hl),cvPoint(crossCenter.x-hl,crossCenter.y+hl),color, thickness, line_type, shift);

	}

	else if(mode == XCV_DRAWCROSS_ASTERRISK)
	{
		int hl=cvRound((double)length/2.0/sqrt(2.0));
		cvLine(dest,cvPoint(crossCenter.x-hl,crossCenter.y-hl),cvPoint(crossCenter.x+hl,crossCenter.y+hl),color, thickness, line_type, shift);
		cvLine(dest,cvPoint(crossCenter.x+hl,crossCenter.y-hl),cvPoint(crossCenter.x-hl,crossCenter.y+hl),color, thickness, line_type, shift);

		hl=length/2;
		cvLine(dest,cvPoint(crossCenter.x-hl,crossCenter.y),cvPoint(crossCenter.x+hl,crossCenter.y),color, thickness, line_type, shift);
		cvLine(dest,cvPoint(crossCenter.x,crossCenter.y-hl),cvPoint(crossCenter.x,crossCenter.y+hl),color, thickness, line_type, shift);

	}
}

void xcvDrawCircles(IplImage* dest, CvMat* points,int radius, CvScalar color,int thickness, int line_type,int shift)
{
	for(int i=0;i<points->rows;i++)
	{
		cvCircle(dest,cvPoint(cvRound(cvmGet(points,i,0)),cvRound(cvmGet(points,i,1))),radius,color,thickness,line_type,shift);
	}
}

void xcvPutText(IplImage* render,char* text,CvPoint orign, CvScalar color, double amp, double shear,int fontType,int thickness)
{
	CvFont font;
	cvInitFont(&font,fontType,amp,amp,shear,thickness);
	cvPutText(render,text,orign,&font, color);
}

class gnuplot
{
	FILE* fp;
public:
	gnuplot(char* gnuplotpath = "C:/fukushima/docs/Dropbox/bin/gnuplot/bin/pgnuplot.exe")
	{
			if((fp = _popen(gnuplotpath,"w"))==NULL)
			{
				fprintf(stderr,"Cannot open gnuplot @ %s\n",gnuplotpath);
				exit(1);
			}
	}
	void cmd(const char* name)
	{
		fprintf(fp,name);
		fflush(fp);
	}
	~gnuplot()
	{
			fclose(fp);
			_pclose(fp);
	}
};

void Plot::point2val(CvPoint pt, double* valx, double* valy)
{
	double x = (double)plotImage->width/(xmax-xmin);
	double y = (double)plotImage->height/(ymax-ymin);
	int H = plotImage->height-1;

	*valx = (pt.x-(origin.x)*2)/x+xmin;
	*valy= (H-(pt.y-origin.y))/y+ymin;
}

void xcvPlotData(IplImage* render,CvMat* data,int size,double xmin,double xmax, double ymin, double ymax,
				  CvScalar color, int lt, int isLine,int thickness, int ps)
{
	double x = (double)render->width/(xmax-xmin);
	double y = (double)render->height/(ymax-ymin);

	int H = render->height-1;

	for(int i=0;i<size;i++)
	{
		double src = cvmGet(data,i,0);
		double dest = cvmGet(data,i,1);

		CvPoint p = cvPoint(cvRound(x*(src-xmin)),H-cvRound(y*(dest-ymin)));
		if(isLine == XCV_LINE_LINEAR)
		{
			if(i!=size-1)
			{
				double nsrc = cvmGet(data,i+1,0);
				double ndest = cvmGet(data,i+1,1);
				cvLine(render,p,cvPoint(cvRound(x*(nsrc-xmin)),H-cvRound(y*(ndest-ymin))),
					color,thickness);
			}
		}
		else if(isLine == XCV_LINE_H2V)
		{
			if(i!=size-1)
			{
				double nsrc = cvmGet(data,i+1,0);
				double ndest = cvmGet(data,i+1,1);
				cvLine(render,p,cvPoint(cvRound(x*(nsrc-xmin)),p.y), color,thickness);
				cvLine(render,cvPoint(cvRound(x*(nsrc-xmin)),p.y),cvPoint(cvRound(x*(nsrc-xmin)),H-cvRound(y*(ndest-ymin))),color,thickness);
			}
		}
		else if(isLine == XCV_LINE_V2H)
		{
			if(i!=size-1)
			{
				double nsrc = cvmGet(data,i+1,0);
				double ndest = cvmGet(data,i+1,1);
				cvLine(render,p,cvPoint(p.x,H-cvRound(y*(ndest-ymin))), color,thickness);
				cvLine(render,cvPoint(p.x,H-cvRound(y*(ndest-ymin))),cvPoint(cvRound(x*(nsrc-xmin)),H-cvRound(y*(ndest-ymin))),color,thickness);
			}
		}

		if(lt==XCV_PLOT_PLUS)
		{
			xcvDrawCross(render,p,2*ps+1,color,XCV_DRAWCROSS_PLUS,thickness);
		}
		else if(lt==XCV_PLOT_CROSS)
		{
			xcvDrawCross(render,p,2*ps+1,color,XCV_DRAWCROSS_TIMES,thickness);
		}
		else if(lt==XCV_PLOT_ASTERRISK)
		{
			xcvDrawCross(render,p,2*ps+1,color,XCV_DRAWCROSS_ASTERRISK,thickness);
		}
		else if(lt==XCV_PLOT_CIRCLE)
		{
			cvCircle(render,p,ps,color,thickness);
		}
		else if(lt==XCV_PLOT_RECTANGLE)
		{
			cvRectangle(render,cvPoint(p.x-ps,p.y-ps),cvPoint(p.x+ps,p.y+ps),
				color,thickness);
		}
		else if(lt==XCV_PLOT_CIRCLE_FILL)
		{
			cvCircle(render,p,ps,color,CV_FILLED);
		}
		else if(lt==XCV_PLOT_RECTANGLE_FILL)
		{
			cvRectangle(render,cvPoint(p.x-ps,p.y-ps),cvPoint(p.x+ps,p.y+ps),
				color,CV_FILLED);
		}
		else if(lt == XCV_PLOT_TRIANGLE)
		{
			xcvTriangle(render,p,2*ps,color,thickness);
		}
		else if(lt == XCV_PLOT_TRIANGLE_FILL)
		{
			xcvTriangle(render,p,2*ps,color,CV_FILLED);
		}
		else if(lt == XCV_PLOT_TRIANGLE_INV)
		{
			xcvTriangleInv(render,p,2*ps,color,thickness);
		}
		else if(lt == XCV_PLOT_TRIANGLE_INV_FILL)
		{
			xcvTriangleInv(render,p,2*ps,color,CV_FILLED);
		}
	}
}

#define XCV_PLOT_BUFFER_SIZE 1024
#define ORIGIN cvPoint(64,64);
void Plot::init()
{

	for(int i=0;i<XCV_PLOTDATA_MAX;i++)
	{
		matsize[i]=XCV_PLOT_BUFFER_SIZE;
		data[i]=NULL;
		data_size[i]=0;
		thickness[i]=1;
		lt[i]=XCV_PLOT_PLUS;
		isLine[i]=XCV_LINE_LINEAR;

		double v=(double)i/data_max*255.0;
		color[i]=getPseudoColor(cv::saturate_cast<uchar>(v));
	}

	color[0]=XCV_RED;
	lt[0]=XCV_PLOT_PLUS;

	color[1]=XCV_GREEN;
	lt[1]=XCV_PLOT_CROSS;

	color[2]=XCV_BLUE;
	lt[2] = XCV_PLOT_ASTERRISK;

	color[3]=XCV_MAGENDA;
	lt[3]=XCV_PLOT_RECTANGLE;

	color[4]=CV_RGB(0,0,128);
	lt[4]=XCV_PLOT_RECTANGLE_FILL;

	color[5]=CV_RGB(128,0,0);
	lt[5]=XCV_PLOT_CIRCLE;

	color[6]=CV_RGB(0,128,128);
	lt[6]=XCV_PLOT_CIRCLE_FILL;

	color[7]=CV_RGB(0,0,0);
	lt[7]=XCV_PLOT_TRIANGLE;

	color[8]=CV_RGB(128,128,128);
	lt[8]=XCV_PLOT_TRIANGLE_FILL;

	color[9]=CV_RGB(0,128,64);
	lt[9]=XCV_PLOT_TRIANGLE_INV;

	color[10]=CV_RGB(128,128,0);
	lt[10]=XCV_PLOT_TRIANGLE_INV_FILL;


	data[0]=cvCreateMat(matsize[0],2,CV_64F);

	setPlotProfile(false,true,false);
	graphImage = render;
}
void Plot::setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_,bool isZeroCross_)
{
	isZeroCross=isZeroCross_;
	isXYMAXMIN = isXYMAXMIN_;
	isXYCenter=isXYCenter_;
}
void Plot::free()
{
	for(int i=0;i<XCV_PLOTDATA_MAX;i++)
	{
		if(data[i]!=NULL)cvReleaseMat(&data[i]);
	}
}
void Plot::setPlotImageSize(Size s)
{
	if(plotImage!=NULL)cvReleaseImage(&plotImage);
	if(render!=NULL)cvReleaseImage(&render);
	plotsize = s;
	plotImage = cvCreateImage(plotsize,8,3);
	render = cvCreateImage(cvSize(plotsize.width+4*origin.x,plotsize.height+2*origin.y),8,3);
}

Plot::Plot(Size plotsize_)
{
	for(int i=0;i<XCV_PLOTDATA_MAX;i++)
	{
		sprintf(keyname[i],"data %02d",i);
	}
	data_max = 1;
	sprintf(xlabel,"x");
	sprintf(ylabel,"y");
	setBackGoundColor(XCV_WHITE);

	origin =ORIGIN;
	plotImage=NULL;
	render = NULL;
	setPlotImageSize(plotsize_);

	keyImage = cvCreateImage(cvSize(256,256),8,3);
	cvSet(keyImage,background_color);

	setXYMinMax(0,plotsize.width,0,plotsize.height);
	isPosition=true;
	init();
}
Plot::~Plot()
{
	free();
	cvReleaseImage(&keyImage);
	cvReleaseImage(&plotImage);
	cvReleaseImage(&render);
}


void Plot::setXYOriginZERO()
{
	recomputeXYMAXMIN(false);
	xmin = 0;
	ymin = 0;
}
void Plot::setYOriginZERO()
{
	recomputeXYMAXMIN(false);
	ymin = 0;
}
void Plot::setXOriginZERO()
{
	recomputeXYMAXMIN(false);
	xmin = 0;
}

void Plot::recomputeXYMAXMIN(bool isCenter, double marginrate)
{
	if(marginrate<0.0 ||marginrate>1.0)marginrate = 1.0;
	xmax = -INT_MAX;
	xmin = INT_MAX;
	ymax = -INT_MAX;
	ymin = INT_MAX;
	for(int i=0;i<data_max;i++)
	{
		for(int j=0;j<data_size[i];j++)
		{
			xmax = (xmax<cvmGet(data[i],j,0))?cvmGet(data[i],j,0):xmax;
			xmin = (xmin>cvmGet(data[i],j,0))?cvmGet(data[i],j,0):xmin;

			ymax = (ymax<cvmGet(data[i],j,1))?cvmGet(data[i],j,1):ymax;
			ymin = (ymin>cvmGet(data[i],j,1))?cvmGet(data[i],j,1):ymin;
		}
	}
	xmax_no_margin = xmax;
	xmin_no_margin = xmin;
	ymax_no_margin = ymax;
	ymin_no_margin = ymin;

	double xmargin  = (xmax-xmin)*(1.0-marginrate)*0.5;
	xmax+=xmargin;
	xmin-=xmargin;

	double ymargin  = (ymax-ymin)*(1.0-marginrate)*0.5;
	ymax+=ymargin;
	ymin-=ymargin;

	if(isCenter)
	{
		double xxx=abs(xmax);
		double yyy=abs(ymax);
		xxx=(xxx<abs(xmin))?abs(xmin):xxx;
		yyy=(yyy<abs(ymin))?abs(ymin):yyy;

		xmax=xxx;
		xmin = -xxx;
		ymax=yyy;
		ymin = -yyy;

		xxx=abs(xmax_no_margin);
		yyy=abs(ymax_no_margin);
		xxx=(xxx<abs(xmin_no_margin))?abs(xmin_no_margin):xxx;
		yyy=(yyy<abs(ymin_no_margin))?abs(ymin_no_margin):yyy;

		xmax_no_margin = xxx;
		xmin_no_margin = -xxx;
		ymax_no_margin =  yyy;
		ymin_no_margin = -yyy;
	}
}
void Plot::setXYMinMax(double xmin_,double xmax_,double ymin_,double ymax_)
{
	xmin=xmin_;
	xmax=xmax_;
	ymin=ymin_;
	ymax=ymax_;

	xmax_no_margin = xmax;
	xmin_no_margin = xmin;
	ymax_no_margin = ymax;
	ymin_no_margin = ymin;
}
void Plot::setXMinMax(double xmin_,double xmax_)
{
	recomputeXYMAXMIN(isXYCenter);
	xmin=xmin_;
	xmax=xmax_;
}
void Plot::setYMinMax(double ymin_,double ymax_)
{
	recomputeXYMAXMIN(isXYCenter);
	ymin=ymin_;
	ymax=ymax_;
}
void Plot::setBackGoundColor(Scalar cl)
{
	background_color=cl;
}
void Plot::setPlot(int plotnum, CvScalar color_,int lt_, int isLine_,int thickness_)
{
	color[plotnum] = color_;
	lt[plotnum]=lt_;
	isLine[plotnum]=isLine_;
	thickness[plotnum]=thickness_;
}

void Plot::setLinetypeALL(int linetype)
{
	for(int i=0;i<XCV_PLOTDATA_MAX;i++)
	{
		isLine[i]=linetype;
	}
}
void Plot::clear(int datanum)
{
	if(datanum<0)
	{
	for(int i=0;i<data_max;i++)
		data_size[i]=0;
	}
	else
		data_size[datanum]=0;

}
void Plot::add(double x, double y, int num)
{
	if(num!=0)
	{
		if(data_max<num+1)
			data_max=num+1;
	}
	if(data[num]==NULL)
	{
		data[num]=cvCreateMat(matsize[num],2,CV_64F);
	}
	//テンプレートとかで実装したほうが．．．
	if(data_size[num]>=matsize[num])
	{
		CvMat* temp = cvCreateMat(matsize[num]*2,2,CV_64F);
		for(int i=0;i<matsize[num];i++)
		{
			cvmSet(temp,i,0,cvmGet(data[num],i,0));
			cvmSet(temp,i,1,cvmGet(data[num],i,1));
		}
		cvReleaseMat(&data[num]);
		data[num]=temp;
		matsize[num]*=2;
	}
	cvmSet(data[num],data_size[num],0,x);
	cvmSet(data[num],data_size[num],1,y);
	data_size[num]++;
}
void Plot::add(vector<cv::Point> point, int num)
{
	//??
	for(int i=0;i<(int)point.size()-1;i++)
	{
		add(point[i].x,point[i].y,num);
	}
}

void Plot::add(vector<cv::Point2d> point, int num)
{
	for(int i=0;i<(int)point.size()-1;i++)
	{
		add(point[i].x,point[i].y,num);
	}
}
void Plot::add(vector<cv::Point2f> point, int num)
{
	for(int i=0;i<(int)point.size()-1;i++)
	{
		add(point[i].x,point[i].y,num);
	}
}

void Plot::makeBB(bool isFont)
{
	
	cvSet(render,background_color);
	cvSetImageROI(render,cvRect(origin.x*2,origin.y,plotsize.width,plotsize.height));
	cvRectangle(plotImage,cvPoint(0,0),cvPoint(plotImage->width-1,plotImage->height-1),XCV_BLACK,1);
	cvCopy(plotImage,render);
	cvResetImageROI(render);

	if(isFont)
	{
		xcvPutText(render,xlabel,cvPoint(render->width/2,(int)(origin.y*1.85+plotImage->height)),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		xcvPutText(render,ylabel,cvPoint(origin.y*1/4,render->height/2),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		
		char buff[128];
		//x coordinate
		sprintf(buff,"%.2f",xmin );
		xcvPutText(render,buff,cvPoint(origin.x,(int)(origin.y*1.35+plotImage->height)),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		
		sprintf(buff,"%.2f",(xmax -xmin )*0.25+xmin );
		xcvPutText(render,buff,cvPoint(origin.x+plotImage->width/4+15,(int)(origin.y*1.35+plotImage->height)),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);

		sprintf(buff,"%.2f",(xmax -xmin )*0.5+xmin );
		xcvPutText(render,buff,cvPoint(origin.x+plotImage->width/2+45,(int)(origin.y*1.35+plotImage->height)),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		
		sprintf(buff,"%.2f",(xmax -xmin )*0.75+xmin );
		xcvPutText(render,buff,cvPoint(origin.x+plotImage->width*3/4+35,(int)(origin.y*1.35+plotImage->height)),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		
		sprintf(buff,"%.2f",xmax );
		xcvPutText(render,buff,cvPoint(plotImage->width+origin.x,(int)(origin.y*1.35+plotImage->height)),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		
		//y coordinate
		sprintf(buff,"%.2f",ymin );
		xcvPutText(render,buff,cvPoint(origin.x,origin.y+plotImage->height),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		
		sprintf(buff,"%.2f",(ymax -ymin )*0.5+ymin );
		xcvPutText(render,buff,cvPoint(origin.x,origin.y+plotImage->height/2),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);

		sprintf(buff,"%.2f",(ymax -ymin )*0.25+ymin );
		xcvPutText(render,buff,cvPoint(origin.x,origin.y+plotImage->height*3/4),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);

		sprintf(buff,"%.2f",(ymax -ymin )*0.75+ymin );
		xcvPutText(render,buff,cvPoint(origin.x,origin.y+plotImage->height/4),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);

		sprintf(buff,"%.2f",ymax );
		xcvPutText(render,buff,cvPoint(origin.x,origin.y),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
		
	}
}
void Plot::plotPoint(Point2d point,CvScalar color_ , int thickness_, int linetype)
{
	CvMat* temp=cvCreateMat(5,2,CV_64F);

	cvmSet(temp,0,0,point.x);
	cvmSet(temp,0,1,ymin);
	cvmSet(temp,1,0,point.x);
	cvmSet(temp,1,1,ymax);
	cvmSet(temp,2,0,point.x);
	cvmSet(temp,2,1,point.y);


	cvmSet(temp,3,0,xmax);
	cvmSet(temp,3,1,point.y);

	cvmSet(temp,4,0,xmin);
	cvmSet(temp,4,1,point.y);
	xcvPlotData(plotImage,temp,5,xmin,xmax,ymin,ymax,color_,XCV_PLOT_NOPOINT,linetype,thickness_);
	cvReleaseMat(&temp);
}


void Plot::plotGrid(int level)
{

	if(level>0)
	{
		plotPoint(Point2d((xmax-xmin)/2.0+xmin,(ymax-ymin)/2.0+ymin),XCV_GRAY150,1);
	}
	if(level>1)
	{
		plotPoint(Point2d((xmax-xmin)*1.0/4.0+xmin,(ymax-ymin)*1.0/4.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*3.0/4.0+xmin,(ymax-ymin)*1.0/4.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*1.0/4.0+xmin,(ymax-ymin)*3.0/4.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*3.0/4.0+xmin,(ymax-ymin)*3.0/4.0+ymin),XCV_GRAY200,1);
	}
	if(level>2)
	{
		plotPoint(Point2d((xmax-xmin)*1.0/8.0+xmin,(ymax-ymin)*1.0/8.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*3.0/8.0+xmin,(ymax-ymin)*1.0/8.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*1.0/8.0+xmin,(ymax-ymin)*3.0/8.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*3.0/8.0+xmin,(ymax-ymin)*3.0/8.0+ymin),XCV_GRAY200,1);

		plotPoint(Point2d((xmax-xmin)*(1.0/8.0+0.5)+xmin,(ymax-ymin)*1.0/8.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(3.0/8.0+0.5)+xmin,(ymax-ymin)*1.0/8.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(1.0/8.0+0.5)+xmin,(ymax-ymin)*3.0/8.0+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(3.0/8.0+0.5)+xmin,(ymax-ymin)*3.0/8.0+ymin),XCV_GRAY200,1);

		plotPoint(Point2d((xmax-xmin)*(1.0/8.0+0.5)+xmin,(ymax-ymin)*(1.0/8.0+0.5)+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(3.0/8.0+0.5)+xmin,(ymax-ymin)*(1.0/8.0+0.5)+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(1.0/8.0+0.5)+xmin,(ymax-ymin)*(3.0/8.0+0.5)+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(3.0/8.0+0.5)+xmin,(ymax-ymin)*(3.0/8.0+0.5)+ymin),XCV_GRAY200,1);

		plotPoint(Point2d((xmax-xmin)*(1.0/8.0)+xmin,(ymax-ymin)*(1.0/8.0+0.5)+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(3.0/8.0)+xmin,(ymax-ymin)*(1.0/8.0+0.5)+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(1.0/8.0)+xmin,(ymax-ymin)*(3.0/8.0+0.5)+ymin),XCV_GRAY200,1);
		plotPoint(Point2d((xmax-xmin)*(3.0/8.0)+xmin,(ymax-ymin)*(3.0/8.0+0.5)+ymin),XCV_GRAY200,1);
	}
}

static void guiPreviewMouse(int event, int x, int y, int flags, void* param)
{
	CvPoint* ret=(CvPoint*)param;

	if(flags == CV_EVENT_FLAG_LBUTTON)
	{
		ret->x=x;
		ret->y=y;
	}
}
void Plot::setKeyName(char* name, int num)
{
	sprintf(keyname[num],"%s",name);
}
void Plot::makeKey(int num)
{
	int step = 20;
	int height = (int)(0.8*keyImage->height);
	CvMat* data = cvCreateMat(2,2,CV_64F);
	for(int i=0;i<num;i++)
	{
		cvmSet(data,0,0,192.0);
		cvmSet(data,0,1,keyImage->height-(i+1)*20);

		cvmSet(data,1,0,keyImage->width-20);
		cvmSet(data,1,1,keyImage->height-(i+1)*20);

		xcvPlotData(keyImage,data,2,0, keyImage->width,0,keyImage->height,color[i],lt[i],isLine[i],thickness[i]);

		xcvPutText(keyImage,keyname[i],cvPoint(0,(i+1)*20+3),color[i],1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL);
	}
}

void Plot::plotData(int gridlevel,int isKey)
{
	cvSet(plotImage,background_color);

	plotGrid(gridlevel);
	if(isZeroCross)	plotPoint(Point2d(0.0,0.0),XCV_ORANGE,1);

	for(int i=0;i<data_max;i++)
	{
		xcvPlotData(plotImage,data[i],data_size[i],xmin, xmax,ymin,ymax,color[i],lt[i],isLine[i],thickness[i]);
	}
	makeBB(true);

	Ptr<IplImage> temp = cvCloneImage(render);
	if(isKey!=0)
		{
			if(isKey==1)
			{
				cvSetImageROI(render,cvRect(render->width-keyImage->width-150,80,keyImage->width,keyImage->height));
			}
			else if(isKey==4)
			{
				cvSetImageROI(render,cvRect(render->width-keyImage->width-150,render->height-keyImage->height-150,keyImage->width,keyImage->height));
			}
			else if(isKey==2)
			{
				cvSetImageROI(render,cvRect(160,80,keyImage->width,keyImage->height));
			}
			else if(isKey==3)
			{
				cvSetImageROI(render,cvRect(160,render->height-keyImage->height-150,keyImage->width,keyImage->height));
			}
			cvCopy(keyImage,render);
			cvResetImageROI(render);
		}
	cvAddWeighted(render,0.8,temp,0.2,0.0,render);
	Mat(render).copyTo(renderMat);
}

void Plot::save(char* name)
{
	FILE* fp = fopen(name,"w");
	
	int dmax = data_size[0];
	for(int i=1;i<data_max;i++)
	{
		dmax = max(data_size[i],dmax);
	}

	for(int n=0;n<dmax;n++)
	{
		for(int i=0;i<data_max;i++)	
		{
			if(n<data_size[i])
			{
				double x = cvmGet(data[i],n,0);
				double y = cvmGet(data[i],n,1);
				fprintf(fp,"%f %f ",x,y);
			}
			else
			{
				double x = cvmGet(data[i],data_size[i]-1,0);
				double y = cvmGet(data[i],data_size[i]-1,1);
				fprintf(fp,"%f %f ",x,y);
			}
		}
		fprintf(fp,"\n");
		//xcvPlotData(plotImage,data[i],data_size[i],xmin, xmax,ymin,ymax,color[i],lt[i],isLine[i],thickness[i]);
	}
	cout<<"p ";
	for(int i=0;i<data_max;i++)
	{
		cout<<"'"<<name<<"'"<<" u "<<2*i+1<<":"<<2*i+2<<" w lp"<<",";
	}
	cout<<endl;
	fclose(fp);
}

Scalar Plot::getPseudoColor(uchar val)
{
	int i=val;
	double d = 255.0/63.0;
	Scalar ret;

	{//g
		uchar lr[256];
		for (int i=0; i<64; i++)
			lr[i]=cvRound(d*i);
		for (int i=64; i<192; i++)
			lr[i]=255;
		for (int i=192; i<256; i++)
			lr[i]=cvRound(255-d*(i-192));

		ret.val[1]=lr[val];
	}
		{//r
		uchar lr[256];
		for (int i=0; i<128; i++)
			lr[i]=0;
		for (int i=128; i<192; i++)
			lr[i]=cvRound(d*(i-128));
		for (int i=192; i<256; i++)
			lr[i]=255;

		ret.val[0]=lr[val];
		}
		{//b
			uchar lr[256];
			for (int i=0; i<64; i++)
				lr[i]=255;
			for (int i=64; i<128; i++)
				lr[i]=cvRound(255-d*(i-64));
			for (int i=128; i<256; i++)
				lr[i]=0;
			ret.val[2]=lr[val];
		}
		return ret;
}


void Plot::plot(char* wname)
{	
	CvPoint pt=cvPoint(0,0);
	cvNamedWindow(wname);

	plotData(0,false);
	//int ym=ymax;
	//int yn=ymin;
	//createTrackbar("ymax",wname,&ym,ymax*2);
	//createTrackbar("ymin",wname,&yn,ymax*2);
	cvSetMouseCallback(wname, (CvMouseCallback)guiPreviewMouse,(void*)&pt );
	int key = 0;
	int isKey=1;
	int gridlevel=0;
	makeKey(data_max);

	recomputeXYMAXMIN();
	while(key!='q')
	{
		//ymax=ym+1;
		//ymin=yn;
		plotData(gridlevel,isKey);
		if(isPosition)
		{
			char text[128];
			double xx = 0.0;
			double yy = 0.0;
			point2val(pt,&xx,&yy);
			if(pt.x<0 ||pt.y<0 ||pt.x>=render->width||pt.y>=render->height)
			{
				pt=cvPoint(0,0);
				xx=0.0;
				yy=0.0;
			}
			sprintf(text,"(%f,%f)",xx,yy);
			xcvPutText(render,text,cvPoint(100,30),XCV_BLACK,1.0,0.0,CV_FONT_HERSHEY_COMPLEX_SMALL,0);
		}

		if(isPosition) xcvDrawGrid(render,pt,CV_RGB(255,180,180),1,4,0);
		cvShowImage(wname,render);
		key = cvWaitKey(1);

		if(key =='?')
		{
			cout<<"*** Help message ***"<<endl;
			cout<<"m: "<<"show mouseover position and grid"<<endl;
			cout<<"c: "<<"(0,0)point must posit center"<<endl;
			cout<<"g: "<<"Show grid"<<endl;
			
			cout<<"k: "<<"Show key"<<endl;
			
			cout<<"x: "<<"Set X origin zero "<<endl;
			cout<<"y: "<<"Set Y origin zero "<<endl;
			cout<<"z: "<<"Set XY origin zero "<<endl;
			cout<<"r: "<<"Reset XY max min"<<endl;
			
			cout<<"s: "<<"Save image (plot.png)"<<endl;
			cout<<"q: "<<"Quit"<<endl;
			
			cout<<"********************"<<endl;
			cout<<endl;
		}
		if(key=='m')
		{
			isPosition=(isPosition)?false:true;
		}

		if(key =='r')
		{
			recomputeXYMAXMIN(false);
		}
		if(key =='c')
		{
			recomputeXYMAXMIN(true);
		}
		if(key =='x')
		{
			setXOriginZERO();
		}
		if(key =='y')
		{
			setYOriginZERO();
		}
		if(key =='z')
		{
			setXYOriginZERO();
		}
		if(key=='k')
		{
			isKey++;
			if(isKey==5)
				isKey=0;
		}
		if(key =='g')
		{
			gridlevel++;
			if(gridlevel>3)gridlevel=0;
		}
		if(key =='p')
		{
			save("plot");
			std::string a("plot ");
			gnuplot gplot;
			for(int i=0;i<data_max;i++)	
			{
				char name[64];
				if(i!=data_max-1)
					sprintf(name,"'plot' u %d:%d w lp,",2*i+1,2*i+2);
				else 
					sprintf(name,"'plot' u %d:%d w lp\n",2*i+1,2*i+2);
				a+=name;
			}
			gplot.cmd(a.c_str());
		}
		if(key =='s')
		{
			save("plot");
			cvSaveImage("plotim.png",render);
			IplImage* mask = cvCreateImage(cvGetSize(plotImage),8,1);
			cvZero(mask);
			IplImage* save = cvCreateImage(cvGetSize(plotImage),8,4);
			IplImage* temp = cvCreateImage(cvGetSize(plotImage),8,3);

			cvCvtColor(plotImage,mask,CV_BGR2GRAY);
			cvCmpS(mask,background_color.val[0],mask,CV_CMP_EQ);
			cvNot(mask,mask);
			//xcvAddAlphaChannel(plotImage,mask,save);
			cvSaveImage("plot.png",save);
			cvReleaseImage(&temp);
			cvReleaseImage(&mask);
			cvReleaseImage(&save);
		}
	}
	cvDestroyWindow(wname);
}

#include "util.h"
#include <stdarg.h>

void rotYaw(Mat& src, Mat& dest, const double yaw)
{
	double angle = yaw/180.0*CV_PI;
	Mat rot = Mat::eye(3,3,CV_64F);

	rot.at<double>(1,1) = cos(angle);
	rot.at<double>(1,2) = sin(angle);
	rot.at<double>(2,1) = -sin(angle);
	rot.at<double>(2,2) = cos(angle);

	Mat a = rot*src;
	a.copyTo(dest);
}

void rotPitch(Mat& src, Mat& dest, const double pitch)
{
	double angle = pitch/180.0*CV_PI;
	Mat rot = Mat::eye(3,3,CV_64F);

	rot.at<double>(0,0) = cos(angle);
	rot.at<double>(0,2) = -sin(angle);
	rot.at<double>(2,0) = sin(angle);
	rot.at<double>(2,2) = cos(angle);

	Mat a = rot*src;
	a.copyTo(dest);
}


void point3d2Mat(const Point3d& src, Mat& dest)
{
	dest.create(3,1,CV_64F);
	dest.at<double>(0,0)=src.x;
	dest.at<double>(1,0)=src.y;
	dest.at<double>(2,0)=src.z;
}

void setXYZ(Mat& in, double&x, double&y, double&z)
{
	x=in.at<double>(0,0);
	y=in.at<double>(1,0);
	z=in.at<double>(2,0);

	//	cout<<format("set XYZ: %.04f %.04f %.04f\n",x,y,z);
}

void eular2rot(double pitch, double roll, double yaw, Mat& dest)
{
	dest = Mat::eye(3,3,CV_64F);
	rotYaw(dest,dest,yaw);
	rotPitch(dest,dest,pitch);
	rotPitch(dest,dest,roll);
}

/*
void lookatBF(const Point3d& from, const Point3d& to, Mat& destR)
{
	double x,y,z;

	Mat fromMat;
	Mat toMat;
	point3d2Mat(from,fromMat);
	point3d2Mat(to,toMat);

	Mat fromtoMat; 
	add(toMat,fromMat,fromtoMat,Mat(),CV_64F);
	double ndiv = 1.0/norm(fromtoMat);
	fromtoMat*=ndiv;

	setXYZ(fromtoMat,x,y,z);
	destR = Mat::eye(3,3,CV_64F);
	double yaw   =-z/abs(z)*asin(y/sqrt(y*y+z*z))/CV_PI*180.0;

	rotYaw(destR,destR,yaw);

	Mat RfromtoMat = destR*fromtoMat;

	setXYZ(RfromtoMat,x,y,z);
	double pitch =z/abs(z)*asin(x/sqrt(x*x+z*z))/CV_PI*180.0;

	rotPitch(destR,destR,pitch);
}
*/
void lookat(const Point3d& from, const Point3d& to, Mat& destR)
{
	Mat destMat = Mat(Point3d(0.0,0.0,1.0));
	Mat srcMat = Mat(from+to);
	srcMat= srcMat/norm(srcMat);

	Mat rotaxis = srcMat.cross(destMat);
	double angle = acos(srcMat.dot(destMat));
	//normalize cross product and multiply rotation angle
	rotaxis=rotaxis/norm(rotaxis)*angle;
	Rodrigues(rotaxis,destR);	
}

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
	createTrackbar("a:0-s1,100-s2","alphaBlend",&a,100);
	int key = 0;
	Mat show;
	while(key!='q')
	{
		addWeighted(s1,1.0-a/100.0,s2,a/100.0,0.0,show);
		imshow("alphaBlend",show);
		key = waitKey(1);
		if(key=='f')
		{
			a = (a!=0) ? 0 : 100;
			setTrackbarPos("a:0-s1,100-s2","alphaBlend",a);
		}
	}
}


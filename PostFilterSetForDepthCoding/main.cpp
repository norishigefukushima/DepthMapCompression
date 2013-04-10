//for OpenNI Capture
//comment out -> image inpout
//else        -> OpenNI with OpenCV input 
//#define CAP_KINECT 1

//for libjpegturbo: faster jpeg encoder
#define JPEG_TURBO 1

#include <opencv2/opencv.hpp>
#include <iostream>
#include "util.h"
#include "filter.h"
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
#pragma comment(lib, "opencv_highgui240.lib")
//#pragma comment(lib, "opencv_haartraining_engine.lib")
//#pragma comment(lib, "opencv_gpu240.lib")
//#pragma comment(lib, "opencv_flann240.lib")
//#pragma comment(lib, "opencv_features2d240.lib")
#pragma comment(lib, "opencv_core240.lib")
#pragma comment(lib, "opencv_contrib240.lib")
#pragma comment(lib, "opencv_calib3d240.lib")
#endif

using namespace cv;
using namespace std;

#define FOCUS 75.0
#define BASELINE 575.0
#define AMP_DISP 2.6
// disp = a * (focal_baseline/depth ) + b;
void disp8U2depth32F(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
{
	if(dest.empty())dest = Mat::zeros(src.size(),CV_32F);
	if(dest.type()!=CV_32F)dest = Mat::zeros(src.size(),CV_32F);
	const int ssesize = src.size().area()/16;
	const int remsize = src.size().area()-16*ssesize;

	uchar* s=src.ptr<uchar>(0);
	float*  d=dest.ptr<float>(0);

	const __m128 maf = _mm_set1_ps(a*focal_baseline);
	const __m128i zeros = _mm_setzero_si128();
	if(b==0.f)
	{
		for(int i=0;i<ssesize;i++)
		{
			__m128i r0 = _mm_load_si128((const __m128i*)(s));

			__m128i r1 = _mm_unpackhi_epi8(r0,zeros);
			r0 = _mm_unpacklo_epi8(r0,zeros);

			__m128i r2 = _mm_unpacklo_epi16(r0,zeros);
			__m128 v1 = _mm_cvtepi32_ps(r2);
			r2 = _mm_unpackhi_epi16(r0,zeros);
			__m128 v2 = _mm_cvtepi32_ps(r2);

			r2 = _mm_unpacklo_epi16(r1,zeros);
			__m128 v3 = _mm_cvtepi32_ps(r2);
			r2 = _mm_unpackhi_epi16(r1,zeros);
			__m128 v4 = _mm_cvtepi32_ps(r2);

			v1 = _mm_div_ps(maf,v1);
			v2 = _mm_div_ps(maf,v2);
			v3 = _mm_div_ps(maf,v3);
			v4 = _mm_div_ps(maf,v4);

			_mm_stream_ps((d),v1);
			_mm_stream_ps((d+4),v2);
			_mm_stream_ps((d+8),v3);
			_mm_stream_ps((d+12),v4);

			s+=16;
			d+=16;
		}
	}
	else 
	{/*
	 const __m128 mb = _mm_set1_ps(b);
	 for(int i=0;i<ssesize;i++)
	 {
	 __m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
	 __m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 8));

	 __m128i r2 = _mm_unpacklo_epi8(r0,zeros);
	 __m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));
	 r2 = _mm_unpackhi_epi8(r0,zeros);
	 __m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));

	 r2 = _mm_unpacklo_epi8(r1,zeros);
	 __m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));
	 r2 = _mm_unpackhi_epi8(r1,zeros);
	 __m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r2,r2), 16));

	 v1 = _mm_add_ps(_mm_div_ps(maf,v1),mb);
	 v2 = _mm_add_ps(_mm_div_ps(maf,v2),mb);
	 v3 = _mm_add_ps(_mm_div_ps(maf,v3),mb);
	 v4 = _mm_add_ps(_mm_div_ps(maf,v4),mb);

	 _mm_stream_ps((d),v1);
	 _mm_stream_ps((d+4),v2);
	 _mm_stream_ps((d+8),v3);
	 _mm_stream_ps((d+12),v4);

	 s+=16;
	 d+=16;
	 }*/
	}

	for(int i=0;i<remsize;i++)
	{
		*d = cvRound(a*focal_baseline / *s + b);
		s++;
		d++;
	}
}

void depth16U2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
{
	if(dest.empty())dest = Mat::zeros(src.size(),CV_8U);
	if(dest.type()!=CV_8U)dest = Mat::zeros(src.size(),CV_8U);
	const int ssesize = src.size().area()/16;
	const int remsize = src.size().area()-16*ssesize;

	ushort* s=src.ptr<ushort>(0);
	uchar*  d=dest.ptr<uchar>(0);

	const __m128 maf = _mm_set1_ps(a*focal_baseline);
	if(b==0.f)
	{
		for(int i=0;i<ssesize;i++)
		{
			__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
			__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

			__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
			__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

			r0 = _mm_loadl_epi64((const __m128i*)(s + 8));
			r1 = _mm_loadl_epi64((const __m128i*)(s + 12));
			__m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
			__m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

			v1 = _mm_div_ps(maf,v1);
			v2 = _mm_div_ps(maf,v2);
			v3 = _mm_div_ps(maf,v3);
			v4 = _mm_div_ps(maf,v4);

			_mm_stream_si128((__m128i*)(d),_mm_packus_epi16(
				_mm_packs_epi32(_mm_cvtps_epi32(v1),_mm_cvtps_epi32(v2)),
				_mm_packs_epi32(_mm_cvtps_epi32(v3),_mm_cvtps_epi32(v4))
				));
			s+=16;
			d+=16;
		}
	}
	else
	{
		const __m128 mb = _mm_set1_ps(b);
		for(int i=0;i<ssesize;i++)
		{
			__m128i r0 = _mm_loadl_epi64((const __m128i*)(s));
			__m128i r1 = _mm_loadl_epi64((const __m128i*)(s + 4));

			__m128 v1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
			__m128 v2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

			r0 = _mm_loadl_epi64((const __m128i*)(s + 8));
			r1 = _mm_loadl_epi64((const __m128i*)(s + 12));
			__m128 v3 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
			__m128 v4 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));

			v1 = _mm_add_ps(_mm_div_ps(maf,v1),mb);
			v2 = _mm_add_ps(_mm_div_ps(maf,v2),mb);
			v3 = _mm_add_ps(_mm_div_ps(maf,v3),mb);
			v4 = _mm_add_ps(_mm_div_ps(maf,v4),mb);

			_mm_stream_si128((__m128i*)(d),_mm_packus_epi16(
				_mm_packs_epi32(_mm_cvtps_epi32(v1),_mm_cvtps_epi32(v2)),
				_mm_packs_epi32(_mm_cvtps_epi32(v3),_mm_cvtps_epi32(v4))
				));
			s+=16;
			d+=16;
		}
	}
	for(int i=0;i<remsize;i++)
	{
		*d = cvRound(a*focal_baseline / *s + b);
		s++;
		d++;
	}
}

void smallGaussianBlur(const Mat& src, Mat& dest, const int d, const double sigma)
{
	if(d==0)
	{
		src.copyTo(dest);
		return ;
	}

	Mat srcf;
	src.convertTo(srcf,CV_32F);
	GaussianBlur(srcf,srcf,Size(d,d),sigma);
	srcf.convertTo(dest,src.type());
}

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

void fillSmallHole(const Mat& src, Mat& dest)
{
	Mat src_;
	if(src.data==dest.data)
		src.copyTo(src_);
	else
		src_ = src;

	uchar* s = (uchar*)src_.ptr<uchar>(1);
	uchar* d = dest.ptr<uchar>(1);
	int step = src.cols*3;
	for(int j=1;j<src.rows-1;j++)
	{
		s+=3,d+=3;
		for(int i=1;i<src.cols-1;i++)
		{
			if(s[1]==0)
			{
				int count=0;
				int b=0,g=0,r=0;

				int lstep;

				lstep = -step -3;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}
				lstep = -step ;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}
				lstep = -step +3;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}
				lstep = -3;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}
				lstep = 3;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}
				lstep = step -3;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}
				lstep = step ;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}
				lstep = step +3;
				if(s[lstep+1 -1] !=0)
				{
					b+=s[lstep+0];
					g+=s[lstep+1];
					r+=s[lstep+2];
					count++;
				}

				d[0]= (count==0) ? 0 : cvRound((double)b/(double)count);
				d[1]= (count==0) ? 0 : cvRound((double)g/(double)count);
				d[2]= (count==0) ? 0 : cvRound((double)r/(double)count);
			}
			s+=3,d+=3;
		}
		s+=3,d+=3;
	}
}

static void myProjectPoint(Point3d& xyz, const Mat& R, const Mat& t, const Mat& K, Point2d& dest)
{
	float r[3][3];
	Mat kr = K*R;
	r[0][0]=(float)kr.at<double>(0,0);
	r[0][1]=(float)kr.at<double>(0,1);
	r[0][2]=(float)kr.at<double>(0,2);

	r[1][0]=(float)kr.at<double>(1,0);
	r[1][1]=(float)kr.at<double>(1,1);
	r[1][2]=(float)kr.at<double>(1,2);

	r[2][0]=(float)kr.at<double>(2,0);
	r[2][1]=(float)kr.at<double>(2,1);
	r[2][2]=(float)kr.at<double>(2,2);

	float tt[3];
	tt[0]=(float)t.at<double>(0,0);
	tt[1]=(float)t.at<double>(1,0);
	tt[2]=(float)t.at<double>(2,0);

	const float x = xyz.x+tt[0];
	const float y = xyz.y+tt[1];
	const float z = xyz.z+tt[2];

	const float div=1.0/( r[2][0]*x + r[2][1]*y + r[2][2]*z);
	dest.x= (r[0][0]*x + r[0][1]*y + r[0][2]*z) * div;
	dest.y= (r[1][0]*x + r[1][1]*y + r[1][2]*z) * div;
}


static void myProjectPoint_SSE(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest)
{
	float r[3][3];
	Mat kr = K*R;
	r[0][0]=(float)kr.at<double>(0,0);
	r[0][1]=(float)kr.at<double>(0,1);
	r[0][2]=(float)kr.at<double>(0,2);

	r[1][0]=(float)kr.at<double>(1,0);
	r[1][1]=(float)kr.at<double>(1,1);
	r[1][2]=(float)kr.at<double>(1,2);

	r[2][0]=(float)kr.at<double>(2,0);
	r[2][1]=(float)kr.at<double>(2,1);
	r[2][2]=(float)kr.at<double>(2,2);

	float tt[3];
	tt[0]=(float)t.at<double>(0,0);
	tt[1]=(float)t.at<double>(1,0);
	tt[2]=(float)t.at<double>(2,0);

	float* data=(float*)xyz.ptr<float>(0);
	Point2f* dst = &dest[0];

	int size1 = (xyz.size().area()/4);
	int size2 = xyz.size().area()%4;

	int i;

	const __m128 addx = _mm_set_ps1(tt[0]);
	const __m128 addy = _mm_set_ps1(tt[1]);
	const __m128 addz = _mm_set_ps1(tt[2]);

	const __m128 r00 = _mm_set_ps1(r[0][0]);
	const __m128 r01 = _mm_set_ps1(r[0][1]);
	const __m128 r02 = _mm_set_ps1(r[0][2]);

	const __m128 r10 = _mm_set_ps1(r[1][0]);
	const __m128 r11 = _mm_set_ps1(r[1][1]);
	const __m128 r12 = _mm_set_ps1(r[1][2]);

	const __m128 r20 = _mm_set_ps1(r[2][0]);
	const __m128 r21 = _mm_set_ps1(r[2][1]);
	const __m128 r22 = _mm_set_ps1(r[2][2]);


	for(i=0;i<size1;i++)
	{
		__m128 a = _mm_load_ps((data));
		__m128 b = _mm_load_ps((data+4));
		__m128 c = _mm_load_ps((data+8));

		__m128 aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(1,2,3,0));
		aa=_mm_blend_ps(aa,b,4);
		__m128 cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(1,3,2,0));
		__m128  mx=_mm_add_ps(addx, _mm_blend_ps(aa,cc,8));

		aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,2,0,1));
		__m128 bb = _mm_shuffle_ps(b,b,_MM_SHUFFLE(2,3,0,1));
		bb=_mm_blend_ps(bb,aa,1);
		cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(2,3,1,0));
		__m128 my=_mm_add_ps(addy,_mm_blend_ps(bb,cc,8));

		aa = _mm_shuffle_ps(a,a,_MM_SHUFFLE(3,1,0,2));
		bb=_mm_blend_ps(aa,b,2);
		cc= _mm_shuffle_ps(c,c,_MM_SHUFFLE(3,0,1,2));
		__m128 mz =_mm_add_ps(addz,_mm_blend_ps(bb,cc,12));

		const __m128 div = _mm_rcp_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(r20,mx),_mm_mul_ps(r21,my)),_mm_mul_ps(r22,mz)));

		a = _mm_mul_ps(div,_mm_add_ps(_mm_add_ps(_mm_mul_ps(r00,mx),_mm_mul_ps(r01,my)),_mm_mul_ps(r02,mz)));
		b = _mm_mul_ps(div,_mm_add_ps(_mm_add_ps(_mm_mul_ps(r10,mx),_mm_mul_ps(r11,my)),_mm_mul_ps(r12,mz)));	

		_mm_stream_ps((float*)dst+4,_mm_unpackhi_ps(a,b));
		_mm_stream_ps((float*)dst,_mm_unpacklo_ps(a,b));

		data+=12;
		dst+=4;
	}
	for(i=0;i<size2;i++)
	{
		const float x = data[0]+tt[0];
		const float y = data[1]+tt[1];
		const float z = data[2]+tt[2];

		const float div=1.0/( r[2][0]*x + r[2][1]*y + r[2][2]*z);

		dst->x= (r[0][0]*x + r[0][1]*y + r[0][2]*z) * div;
		dst->y= (r[1][0]*x + r[1][1]*y + r[1][2]*z) * div;

		data+=3;
		dst++;
	}
}

static void myProjectPoint_BF(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest)
{
	float r[3][3];
	Mat kr = K*R;
	r[0][0]=(float)kr.at<double>(0,0);
	r[0][1]=(float)kr.at<double>(0,1);
	r[0][2]=(float)kr.at<double>(0,2);

	r[1][0]=(float)kr.at<double>(1,0);
	r[1][1]=(float)kr.at<double>(1,1);
	r[1][2]=(float)kr.at<double>(1,2);

	r[2][0]=(float)kr.at<double>(2,0);
	r[2][1]=(float)kr.at<double>(2,1);
	r[2][2]=(float)kr.at<double>(2,2);

	float tt[3];
	tt[0]=(float)t.at<double>(0,0);
	tt[1]=(float)t.at<double>(1,0);
	tt[2]=(float)t.at<double>(2,0);

	float* data=(float*)xyz.ptr<float>(0);
	Point2f* dst = &dest[0];

	int size2 = xyz.size().area();

	int i;
	for(i=0;i<size2;i++)
	{
		const float x = data[0]+tt[0];
		const float y = data[1]+tt[1];
		const float z = data[2]+tt[2];

		const float div=1.0/( r[2][0]*x + r[2][1]*y + r[2][2]*z);

		dst->x= (r[0][0]*x + r[0][1]*y + r[0][2]*z) * div;
		dst->y= (r[1][0]*x + r[1][1]*y + r[1][2]*z) * div;

		data+=3;
		dst++;
	}
}


static void myProjectPoint(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest)
{
	myProjectPoint_SSE(xyz, R, t, K, dest);//SSE implimentation
	//myProjectPoint_BF(xyz, R, t, K, dest);//normal implementation
}

static void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub, vector<Point2f>& pt, Mat& depth)
{
	if(destimage.empty())destimage=Mat::zeros(Size(image.size()),image.type());
	else destimage.setTo(0);
	CV_Assert(
		K.type()==CV_64F &&
		R.type()==CV_64F && 
		t.type()==CV_64F &&
		" only support 64F matrix type");
	{
#ifdef _CALC_TIME_
		CalcTime t1("depth projection to other viewpoint");
#endif
		myProjectPoint(xyz,R,t,K,pt);
	}
#ifdef _CALC_TIME_
	CalcTime tm("rendering");
#endif
	depth.setTo(10000.f);
	//#pragma omp parallel for
	Point2f* ptxy = &pt[0];
	float* xyzdata  = (float*)xyz.ptr<float>(0);
	uchar* img=(uchar*)image.ptr<uchar>(0);

	float* zbuff;
	const int step1 = image.cols;
	const int step3 = image.cols*3;
	const int wstep = destimage.cols*3;

	ptxy+=step1;
	xyzdata+=step3;
	img+=step3;
	for(int j=1;j<image.rows-1;j++)
	{
		ptxy++,xyzdata+=3,img+=3;
		for(int i=1;i<image.cols-1;i++)
		{
			int x=(int)(ptxy->x);
			int y=(int)(ptxy->y);

			//if(m[i]==255)continue;
			if(x>=1 && x<image.cols-1 && y>=1 && y<image.rows-1)
			{
				zbuff = depth.ptr<float>(y)+x;
				const float z =xyzdata[2];

				//cout<<format("%d %d %d %d %d %d \n",j,y, (int)ptxy[image.cols].y,i,x,(int)ptxy[1].x);
				//	getchar();
				if(*zbuff>z)
				{
					uchar* dst = destimage.data + wstep*y + 3*x;
					dst[0]=img[0];
					dst[1]=img[1];
					dst[2]=img[2];
					*zbuff=z;

					if(isSub)
					{
						if((int)ptxy[image.cols].y-y>1 && (int)ptxy[1].x-x>1)
						{
							if(zbuff[1]>z)
							{
								dst[3]=img[0];
								dst[4]=img[1];
								dst[5]=img[2];
								zbuff[1]=z;
							}
							if(zbuff[step1+1]>z)
							{
								dst[wstep+0]=img[0];
								dst[wstep+1]=img[1];
								dst[wstep+2]=img[2];
								zbuff[step1+1]=z;
							}
							if(zbuff[step1]>z)
							{
								dst[wstep+3]=img[0];
								dst[wstep+4]=img[1];
								dst[wstep+5]=img[2];
								zbuff[step1]=z;
							}
						}
						else if((int)ptxy[1].x-x>1)
						{
							if(zbuff[1]>z)
							{
								dst[3]=img[0];
								dst[4]=img[1];
								dst[5]=img[2];
								zbuff[1]=z;
							}
						}
						else if((int)ptxy[image.cols].y-y>1)
						{
							if(zbuff[step1]>z)
							{
								dst[wstep+0]=img[0];
								dst[wstep+1]=img[1];
								dst[wstep+2]=img[2];
								zbuff[step1]=z;
							}
						}

						if((int)ptxy[-image.cols].y-y<-1 && (int)ptxy[-1].x-x<-1)
						{
							if(zbuff[-1]>z)
							{
								dst[-3]=img[0];
								dst[-2]=img[1];
								dst[-1]=img[2];
								zbuff[-1]=z;
							}
							if(zbuff[-step1-1]>z)
							{
								dst[-wstep+0]=img[0];
								dst[-wstep+1]=img[1];
								dst[-wstep+2]=img[2];
								zbuff[-step1-1]=z;
							}
							if(zbuff[-step1]>z)
							{
								dst[-wstep-3]=img[0];
								dst[-wstep-2]=img[1];
								dst[-wstep-1]=img[2];
								zbuff[-step1]=z;
							}
						}
						else if((int)ptxy[-1].x-x<-1)
						{
							if(zbuff[-1]>z)
							{
								dst[-3]=img[0];
								dst[-2]=img[1];
								dst[-1]=img[2];
								zbuff[-1]=z;
							}
						}
						else if((int)ptxy[-image.cols].y-y<-1)
						{
							if(zbuff[-step1]>z)
							{
								dst[-wstep+0]=img[0];
								dst[-wstep+1]=img[1];
								dst[-wstep+2]=img[2];
								zbuff[-step1]=z;
							}
						}
					}
				}
			}
			ptxy++,xyzdata+=3,img+=3;
		}
		ptxy++,xyzdata+=3,img+=3;
	}
}

static void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub)
{
	vector<Point2f> pt(image.size().area());
	Mat depth = 10000.f*Mat::ones(image.size(),CV_32F);

	projectImagefromXYZ(image, destimage, xyz, R, t, K, dist, mask, isSub,  pt, depth);
}

template <class T>
static void fillOcclusion_(Mat& src, const T invalidvalue, const T maxval)
{
	const int MAX_LENGTH=(int)(src.cols*0.5);
	//#pragma omp parallel for
	for(int j=0;j<src.rows;j++)
	{
		T* s = src.ptr<T>(j);

		s[0]=maxval;
		s[src.cols-1]=maxval;

		for(int i=1;i<src.cols-1;i++)
		{
			if(s[i]==invalidvalue)
			{
				int t=i;
				do
				{
					t++;
					if(t>src.cols-1)break;
				}while(s[t]==invalidvalue);

				const T dd = min(s[i-1],s[t]);
				if(t-i>MAX_LENGTH)
				{
					for(int n=0;n<src.cols;n++)
					{
						s[n]=invalidvalue;
					}
				}
				else
				{
					for(;i<t;i++)
					{
						s[i]=dd;
					}
				}
			}
		}
		s[0]=s[1];
		s[src.cols-1]=s[src.cols-2];
	}
}

template <class T>
static void fillOcclusionInv_(Mat& src, const T invalidvalue, const T minval)
{
	const int MAX_LENGTH=(int)(src.cols);
	//#pragma omp parallel for
	for(int j=0;j<src.rows;j++)
	{
		T* s = src.ptr<T>(j);

		s[0]=minval;
		s[src.cols-1]=minval;

		for(int i=1;i<src.cols-1;i++)
		{
			if(s[i]==invalidvalue)
			{
				int t=i;
				do
				{
					t++;
					if(t>src.cols-1)break;
				}while(s[t]==invalidvalue);

				const T dd = max(s[i-1],s[t]);
				if(t-i>MAX_LENGTH)
				{
					for(int n=0;n<src.cols;n++)
					{
						s[n]=invalidvalue;
					}
				}
				else
				{
					for(;i<t;i++)
					{
						s[i]=dd;
					}
				}
			}
		}
		s[0]=s[1];
		s[src.cols-1]=s[src.cols-2];
	}

}

class KinectDepth
{
public:
	Size size;
	Mat depth;
	Mat compressedDepth;
	Mat xyz;


	void getDepth8U(Mat& dest, double div=-1.0)
	{
		if(div<0)
		{
			double minv,maxv;
			minMaxLoc(depth, &minv,&maxv);
			depth.convertTo(dest,CV_8U,255.0/maxv);
		}
		else
		{
			depth.convertTo(dest,CV_8U,255.0/div);
		}
	}
	KinectDepth(Size size_, string name, bool isFillOcclusion=true)
	{
		read(size_,name);
	}
	KinectDepth(Mat& src, bool isFillOcclusion=true)
	{
		if(src.type()==CV_16U)
			src.copyTo(depth);

		size = src.size();

		if(isFillOcclusion)fillOcclusionInv_<ushort>(depth,0,0);
	}
	bool read(Size size_, string name, bool isFillOcclusion=true)
	{
		size =size_;
		depth.create(size,CV_16U);
		FILE* fp = fopen(name.c_str(),"rb");
		if(fp==NULL)	
		{
			cout<<name<<"is not found"<<endl;
			return false;
		}
		unsigned short*buff = depth.ptr<unsigned short>(0);
		fread(buff,sizeof(unsigned short),size.area(),fp);
		fclose(fp);
		if(isFillOcclusion)fillOcclusionInv_<ushort>(depth,0,0);
		return true;
	}
	void write(String name)
	{
		unsigned short* data=depth.ptr<unsigned short>(0);
		FILE* fp = fopen(name.c_str(),"wb");
		fwrite(data,sizeof(unsigned short),size.area(),fp);
		fclose(fp);
	}


	Mat disparityMap;
	void cvtDepth8UC3(Mat& src, Mat& dest)
	{
		unsigned short* dep = src.ptr<unsigned short>(0);
		unsigned char* dst = dest.ptr<unsigned char>(0);
		const float v=510.0*75*3.4;
		dest.setTo(0);
#pragma omp parallel for
		for(int i=0;i<depth.size().area();i++)
		{
			if(dep[i]!=0) 
			{
				uchar vv = (unsigned char)(v/dep[i]);
				dst[3*i+0]=vv;
				dst[3*i+1]=vv;
				dst[3*i+2]=vv;
			}
		}
	}
	void cvtDepth16U(Mat& src, Mat& dest)
	{
		unsigned char* dep = src.ptr<unsigned char>(0);
		unsigned short* dst = dest.ptr<unsigned short>(0);
		const float v=510.0*75*3.4;
		dest.setTo(0);
#pragma omp parallel for
		for(int i=0;i<src.size().area();i++)
		{
			if(dep[i]!=0) 
			{
				dst[i] = (unsigned short)(v/dep[i]);
			}
		}
	}
	void getDisparity8U(Mat& dest)
	{
		if(dest.empty())dest.create(depth.size(),CV_8U);
		unsigned short* dep = depth.ptr<unsigned short>(0);
		unsigned char* dst = dest.ptr<unsigned char>(0);
		const float v=510.0*75*3.4;
		dest.setTo(0);
#pragma omp parallel for
		for(int i=0;i<depth.size().area();i++)
		{
			if(dep[i]!=0) 
			{
				dst[i] = (unsigned char)(v/dep[i]);
			}
		}
	}

	void moveXYZ(Mat& dest, Mat t, Mat R)
	{
		dest.create(depth.size().area(),1,CV_32FC3);

		float r[3][3];
		r[0][0]=(float)R.at<double>(0,0);
		r[0][1]=(float)R.at<double>(0,1);
		r[0][2]=(float)R.at<double>(0,2);

		r[1][0]=(float)R.at<double>(1,0);
		r[1][1]=(float)R.at<double>(1,1);
		r[1][2]=(float)R.at<double>(1,2);

		r[2][0]=(float)R.at<double>(2,0);
		r[2][1]=(float)R.at<double>(2,1);
		r[2][2]=(float)R.at<double>(2,2);

		float tt[3];
		tt[0]=(float)t.at<double>(0,0);
		tt[1]=(float)t.at<double>(1,0);
		tt[2]=(float)t.at<double>(2,0);


#pragma omp parallel for
		for(int j=0;j<depth.rows;j++)
		{
			const int index=j*depth.cols;
			float* data=xyz.ptr<float>(index);
			float* out=dest.ptr<float>(index);
			for(int i=0;i<depth.cols;i++)
			{
				const float x = data[0]+tt[0];
				const float y = data[1]+tt[1];
				const float z = data[2]+tt[2];

				out[0]= r[0][0]*x + r[0][1]*y + r[0][2]*z;
				out[1]= r[1][0]*x + r[1][1]*y + r[1][2]*z;
				out[2]= r[2][0]*x + r[2][1]*y + r[2][2]*z;

				data+=3;
				out+=3;
			}
		}
	}

	Point3d getxyz(Point pt)
	{	
		if(xyz.empty())cout<<"please call reprojectXYZ function"<<endl;
		const int step = size.width*3;
		float* data=xyz.ptr<float>(0);
		data+=pt.y*step+pt.x*3;
		return (Point3d((double)data[0],(double)data[1],(double)data[2]));
	}
	void getXYZLine(vector<Point3f>& dest, int line_height)
	{
		if(xyz.empty())cout<<"please call reprojectXYZ function"<<endl;

		dest.clear();
		float* data=xyz.ptr<float>(line_height*size.width);
		for(int i=0;i<size.width;i++)
		{
			dest.push_back(Point3f(data[0],data[1],data[2]));
			data+=3;
		}
	}

	void reprojectXYZ(double f)
	{
		if(xyz.empty())xyz=Mat::zeros(depth.size().area(),1,CV_32FC3);

		const float bigZ = 10000.f;
		const float fxinv = (float)(1.0/f);
		const float fyinv = (float)(1.0/f);
		const float cw = (depth.size().width-1)*0.5;
		const float ch = (depth.size().height-1)*0.5;

		unsigned short* dep = depth.ptr<unsigned short>(0);
		float* data=xyz.ptr<float>(0);
		for(int j=0;j<depth.rows;j++)
		{
			float b = j-ch;
			const float y = b*fyinv;

			float x = (-cw)*fxinv;
			for(int i=0;i<depth.cols;i++)
			{
				float z = *dep;
				data[0]=x*z;
				data[1]=y*z;
				data[2]= (z==0) ?bigZ:z;

				data+=3,dep++;
				x+=fxinv;
			}
		}
	}

	void reprojectXYZ(Mat& intrinsic, Mat& distortion, float a=1.0, float b=0.0)
	{
		if(xyz.empty())xyz=Mat::zeros(depth.size().area(),1,CV_32FC3);

		const float bigZ = 10000.f;
		const float fxinv = (float)(1.0/intrinsic.at<double>(0,0));
		const float fyinv = (float)(1.0/intrinsic.at<double>(1,1));
		const float cw = (float)intrinsic.at<double>(0,2);
		const float ch = (float)intrinsic.at<double>(1,2);
		const float k0 = distortion.at<double>(0.0);
		const float k1 = distortion.at<double>(1.0);
		//#pragma omp parallel for
		for(int j=0;j<depth.rows;j++)
		{
			const float y = (j-ch)*fyinv;
			const float yy=y*y;
			unsigned short* dep = depth.ptr<unsigned short>(j);
			float* data=xyz.ptr<float>(j*depth.cols);
			for(int i=0;i<depth.cols;i++,dep++,data+=3)
			{
				const float x = (i-cw)*fxinv;
				const float rr = x*x+yy;

				float i2= (k0*rr + k1*rr*rr+1)*i;
				float j2= (k0*rr + k1*rr*rr+1)*j;

				float z = a* *dep+b;
				data[0]=(i2-cw)*fxinv*z;
				data[1]=(j2-ch)*fyinv*z;
				data[2]= (z==0) ?bigZ:z;
			}
		}
	}
	Point3f get3DPoint(double u, double v, Mat& intrinsic, Mat& distortion, double a=1.0, double b=0.0)
	{
		//parameter_a=a;
		//parameter_b=b;

		int iu = (int)u;
		int iv = (int)v;
		double aa = u-iu;
		double bb = v-iv;

		const unsigned short z0 = depth.at<unsigned short>(iv,iu);
		const unsigned short z1 = depth.at<unsigned short>(iv,iu+1);
		const unsigned short z2 = depth.at<unsigned short>(iv+1,iu);
		const unsigned short z3 = depth.at<unsigned short>(iv+1,iu+1);

		double d0 = (1.0-aa)*(1.0-bb)*z0;
		double d1 = (aa)*(1.0-bb)*z1;
		double d2 = (1.0-aa)*(bb)*z2;
		double d3 = (aa)*(bb)*z3;

		double d = d0+d1+d2+d3;
		float z = (float)(a*d+b);
		//cout<<"z: "<<z<<endl;
		if(z0==0||z1==0||z2==0||z3==0)return Point3f(0.f,0.f,0.f);

		vector<Point2f> in;

		in.push_back(Point2f(u,v));
		vector<Point2f> out;

		undistortPoints(in,out,intrinsic,distortion,Mat(),intrinsic);
		//undistortPoints(in,out,Mat::eye(3,3,CV_64F),Mat());

		const double fxinv = 1.0/intrinsic.at<double>(0,0);
		const double fyinv = 1.0/intrinsic.at<double>(1,1);

		//cout<<"(fx,fy): "<<fxinv<<","<<fyinv<<endl;
		const float dcx = out[0].x-intrinsic.at<double>(0,2);
		const float dcy = out[0].y-intrinsic.at<double>(1,2);

		float x = (float)(dcx*fxinv*z);
		float y = (float)(dcy*fyinv*z);

		return Point3f(x,y,z);
	}	
};


static void onMouse(int event, int x, int y, int flags, void* param)
{
	Point* ret=(Point*)param;

	if(flags == CV_EVENT_FLAG_LBUTTON)
	{
		ret->x=x;
		ret->y=y;
	}
}

void projectTest(Mat& image, Mat& srcDepth16)
{	
	namedWindow("image");
	Point pt = Point(image.cols/2,image.rows/2);
	cv::setMouseCallback("image", (MouseCallback)onMouse,(void*)&pt );

	bool isH264=false;
	int q = 80;
	createTrackbar("JPEG q/PNG","image",&q,102);

	int mr = 1;
	createTrackbar("md radius","image",&mr,10);
	int gr = 0;
	createTrackbar("Gauss radius","image",&gr,10);
	int br = 1;
	createTrackbar("br radius","image",&br,10);
	int dr = 3;
	createTrackbar("wbd radius","image",&dr,10);
	int thresh = 65;
	createTrackbar("th","image",&thresh,255);

	int rrr =1;
	createTrackbar("post med radius","image",&rrr,10);

	const int xmax = 8000;
	const int ymax = 8000;
	const int initx = xmax/2;
	const int inity = ymax/2;
	const int initz = 4000;
	const int initpitch = 90;
	const int inityaw = 90;

	int x = initx;
	createTrackbar("x","image",&x,xmax);
	int y = initx;
	createTrackbar("y","image",&y,ymax);
	int z = initz;
	createTrackbar("z","image",&z,8000);

	int pitch = initpitch;
	createTrackbar("pitch","image",&pitch,180);
	int yaw = inityaw;
	createTrackbar("yaw","image",&yaw,180);

	int px = 320;
	createTrackbar("look at x","image",&px,639);
	int py = 240;
	createTrackbar("look at y","image",&py,479);
	int sub=2;
	createTrackbar("render Opt","image",&sub,3);
	int sw = 0;
	createTrackbar("sw","image",&sw,2);

	Mat destImage(Size(640,480),CV_8UC3);	//rendered image
	Mat view;//drawed image

	KinectDepth kd(srcDepth16,true);

	float focal_length=510.f;
	Mat k = Mat::eye(3,3,CV_64F)*focal_length;
	k.at<double>(0,2)=(image.cols-1)*0.5;
	k.at<double>(1,2)=(image.rows-1)*0.5;
	k.at<double>(2,2)=1.0;

	kd.reprojectXYZ(focal_length);

	Point3d srcview;
	Point3d look = kd.getxyz(Point(px,py));

	int count=0;
	bool isDrawLine = true;
	bool isWrite=false;
	bool isLookat = false;


	int postFilterMethod = 2;
	PostFilterSet pfs;

	CalcTime tm("total");
	int key=0;
	Mat srcDepth;

#ifdef CAP_KINECT
	cv::VideoCapture vc(CV_CAP_OPENNI);
#endif

	vector<Point> viewbuff;
	for(int i=0;i<128;i++)
	{
	int r = 30;
	Point v = Point(r*cos(i/20.0)+320-r,r*sin(i/20.0)+240);
	
	viewbuff.push_back(v);
	}

	int cc=0, video=0;
	while(key!='q')
	{
		if(cc!=0)
		{
			pt = viewbuff[cc++];
			if(cc==128)
			{
			cc = 0;
			pt = Point(320,240);
			}
			video++;
			if(video==127){sw=2;cc++;}
			if(video==254)
			{
				video=0;
				sw=0;
			}	
		}
		double bps;
		//from mouse input
		x = (int)(xmax*(double)pt.x/(double)image.cols + 0.5);
		y = (int)(ymax*(double)pt.y/(double)image.rows + 0.5);
		setTrackbarPos("x","image",x);
		setTrackbarPos("y","image",y);

		tm.start();
#ifdef CAP_KINECT
		vc.grab();
		vc.retrieve(srcDepth16,CV_CAP_OPENNI_DEPTH_MAP);
		vc.retrieve(image,CV_CAP_OPENNI_BGR_IMAGE);
#endif
		//compress depth map by JPEG or PNG
		Mat disp;
		if(q==101)// 8bit png
		{
			depth16U2disp8U(srcDepth16, srcDepth, FOCUS*BASELINE, 2.6f);

			fillOcclusion_<uchar>(srcDepth,0,255);
			Mat tr=srcDepth.t();
			fillOcclusion_<uchar>(tr,0,255);
			transpose(tr,srcDepth);

			if(isH264)
			{
				double bpp;
				int size;
				degradeImagex264(srcDepth, disp, 0, size, bpp);
				bps = 30*8.0*size/1000000.0;
			}
			else
			{
				vector<uchar> buff;
				vector<int> param(4);
				param[0] = CV_IMWRITE_PNG_COMPRESSION;
				param[1] = 9;
				param[2] = CV_IMWRITE_PNG_STRATEGY;
				param[3] = CV_IMWRITE_PNG_STRATEGY_RLE;
				imencode(".png",srcDepth,buff,param);
				disp = imdecode(buff,0);
				bps = 30*8.0*buff.size()/1000000.0;
			}
		}
		else if(q==102)//16bit png
		{
			vector<uchar> buff;
			vector<int> param(4);
			param[0] = CV_IMWRITE_PNG_COMPRESSION;
			param[1] = 9;
			param[2] = CV_IMWRITE_PNG_STRATEGY;
			param[3] = CV_IMWRITE_PNG_STRATEGY_RLE;
			fillOcclusionInv_<ushort>(srcDepth16,0,0);
			Mat tr=srcDepth16.t();
			fillOcclusionInv_<ushort>(tr,0,0);
			transpose(tr,srcDepth16);
			srcDepth16.copyTo(kd.depth);
			depth16U2disp8U(srcDepth16, srcDepth, FOCUS*BASELINE, 2.6f);
			imencode(".png",kd.depth,buff,param);
			srcDepth.copyTo(disp);
			bps = 30*8.0*buff.size()/1000000.0;

		}
		else//8 but JPEG
		{	
			depth16U2disp8U(srcDepth16, srcDepth, FOCUS*BASELINE, 2.6f);

			fillOcclusion_<uchar>(srcDepth,0,255);
			Mat tr=srcDepth.t();
			fillOcclusion_<uchar>(tr,0,255);
			transpose(tr,srcDepth);

			if(isH264)
			{
				
				int qp = (50- q/2) + 1;
				double bpp;
				int size;
				degradeImagex264(srcDepth, disp, qp, size, bpp);
				bps = 30*8.0*size/1000000.0;
			}
			else
			{
#ifdef JPEG_TURBO
				int size;
				double bpp;
				degradeJPEG(srcDepth,disp,q,0,true,size,bpp);
				bps = 30*8.0*size/1000000.0;
#else			
				vector<uchar> buff;
				vector<int> param(2);
				param[0] = CV_IMWRITE_JPEG_QUALITY;
				param[1] = q;
				imencode(".jpg",srcDepth,buff,param);
				disp = imdecode(buff,0);
				bps = 30*8.0*buff.size()/1000000.0;
#endif
				
			}
		}

		//post filter set for coded depth map
		Mat dshow;
		if(q==102)//case raw
		{
			srcDepth16.copyTo(kd.depth);
			depth16U2disp8U(kd.depth, dshow, FOCUS*BASELINE, 2.6f,0.f);
		}
		else// case compression
		{
			//CalcTime t("post filter");
			if(postFilterMethod==2)
			{
				pfs.filterDisp8U2Depth16U(disp,kd.depth,FOCUS,BASELINE,AMP_DISP,mr,gr,br,dr,thresh,BILATERAL_NORMAL);
				depth16U2disp8U(kd.depth, dshow, FOCUS*BASELINE, 2.6f,0.f);
			}
			else if(postFilterMethod==1)
			{
				boundaryReconstructionFilter(disp,disp,Size(13,13),1.0,1.0,1.0);
				Mat temp;
				disp8U2depth32F(disp,temp,FOCUS*BASELINE,AMP_DISP,0.f);
				temp.convertTo(kd.depth,CV_16U);
				depth16U2disp8U(kd.depth, dshow, FOCUS*BASELINE, 2.6f,0.f);
			}
			else
			{
				Mat temp;
				disp8U2depth32F(disp,temp,FOCUS*BASELINE,AMP_DISP,0.f);
				temp.convertTo(kd.depth,CV_16U);
				depth16U2disp8U(kd.depth, dshow, FOCUS*BASELINE, 2.6f,0.f);
			}
		}

		{
			//CalcTime t(" depth2xyz projection");
			kd.reprojectXYZ(focal_length);
		}

		Mat R = Mat::eye(3,3,CV_64F);
		Mat t = Mat::zeros(3,1,CV_64F);
		t.at<double>(0,0)=x-initx;
		t.at<double>(1,0)=y-inity;
		t.at<double>(2,0)=-z+initz;

		srcview = Point3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));
		if(isLookat)
		{
			look = kd.getxyz(Point(px,py));
		}
		lookat(look,srcview,R);
		Mat r;
		eular2rot(pitch-90.0,0.0,yaw - 90,r);
		R = r*R;

		//project 3D point image
		if(sw==0)//image view
		{
			if(sub>0)
				projectImagefromXYZ(image,destImage,kd.xyz,R,t,k,Mat(),Mat(),true);
			else
				projectImagefromXYZ(image,destImage,kd.xyz,R,t,k,Mat(),Mat(),false);
		}
		else//depth map view
		{
			
			Mat dispC;
			if(sw==1)
				cvtColor(dshow,dispC,CV_GRAY2BGR);
			else
				cv::applyColorMap(dshow,dispC,2);

			if(sub>0)
				projectImagefromXYZ(dispC,destImage,kd.xyz,R,t,k,Mat(),Mat(),true);
			else
				projectImagefromXYZ(dispC,destImage,kd.xyz,R,t,k,Mat(),Mat(),false);
		}

		//post filter for rendering image
		if(sub>2)fillSmallHole(destImage,destImage);
		if(sub>1)
		{
			Mat gray,mask;
			cvtColor(destImage,gray,CV_BGR2GRAY);
			compare(gray,0,mask,cv::CMP_EQ);
			medianBlur(destImage,view,2*rrr+1);
			destImage.copyTo(view,~mask);
		}
		else destImage.copyTo(view);

		if(cc!=0)imwrite(format("video/im%03d.png",video),view);
		if(isWrite)imwrite(format("out%4d.jpg",count++),view);
		
		if(isDrawLine)
		{
			Point2d ptf;
			myProjectPoint(look,R,t,k,ptf);
			circle(view,Point(ptf),7,CV_RGB(0,255,0),CV_FILLED);
			line(view,Point(0,240),Point(640,240),CV_RGB(255,0,0));
			line(view,Point(320,0),Point(320,480),CV_RGB(255,0,0));
		}
		double fps = 1000.0/tm.getTime();
		putText(view,format("%.02f fps",fps),Point(30,30),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
		if(q<101)
		{
			if(isH264) 
			{
				int qp = (50- q/2) + 1;
				putText(view,format("%.02f Mbps: H.264:qp %02d",bps,qp),Point(30,70),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
			}
			else 
			{
				putText(view,format("%.02f Mbps: JPG: q %02d",bps,q),Point(30,70),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
				
			}
		}
		else 
		{
			if(isH264&&q==101) putText(view,format("%.02f Mbps: H.264:qp 0",bps),Point(30,70),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
			else
				putText(view,format("%.02f Mbps: PNG",bps),Point(30,70),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
		}
		if(postFilterMethod==2)
			putText(view,format("Post Filter: Prop",bps),Point(30,110),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
		else if(postFilterMethod==1)
			putText(view,format("Post Filter: BRF",bps),Point(30,110),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
		else
			putText(view,format("Post Filter: Off",bps),Point(30,110),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
		if(isLookat)
			putText(view,format("Look at: Free",bps),Point(30,150),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));
		else
			putText(view,format("Look at: Fix",bps),Point(30,150),CV_FONT_HERSHEY_DUPLEX,1.0,CV_RGB(255,255,255));

		//show image
		imshow("image",view);
		key = waitKey(1);	

		if(key=='h')
		{
			isH264 = isH264 ? false:true;
		}
		if(key=='g')
		{
			isDrawLine = isDrawLine ? false : true;
		}
		if(key=='l')
		{
			isLookat = (isLookat) ? false : true;
		}
		if(key=='s')
		{
			isWrite = true;
		}

		if(key=='e')
		{
			cc=1;
		}
		if(key=='r')
		{
			pt.x = image.cols/2;
			pt.y = image.rows/2;

			z=initz;
			pitch = initpitch;
			yaw = inityaw;

			setTrackbarPos("z","image",z);
			setTrackbarPos("pitch","image",pitch);
			setTrackbarPos("yaw","image",yaw);
		}
		if(key == 'p')
		{
			postFilterMethod++;
			if(postFilterMethod>2) postFilterMethod=0;
		}
		if(key == 'v')
		{
			sw++;
			if(sw>2) sw=0;
			setTrackbarPos("sw","image",sw);

		}
	}
}

int main(int argc, char** argv)
{
	//Mat src = imread("dataset/kinect/desk_1_1.png");
	//Mat depth16 = imread("dataset/kinect/desk_1_1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
	
	Mat src = imread("dataset/kinect/meeting_small_1_1.png");
	Mat depth16 = imread("dataset/kinect/meeting_small_1_1_depth.png",CV_LOAD_IMAGE_UNCHANGED);

	projectTest(src,depth16);
}
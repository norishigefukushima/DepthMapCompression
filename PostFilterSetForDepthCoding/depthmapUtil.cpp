#include "config.h"
#include "filter.h"

#if CV_SSE4_1
#include <smmintrin.h>
#endif

//point cloud rendering
#if CV_SSE4_1
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

		const float div=1.f/( r[2][0]*x + r[2][1]*y + r[2][2]*z);

		dst->x= (r[0][0]*x + r[0][1]*y + r[0][2]*z) * div;
		dst->y= (r[1][0]*x + r[1][1]*y + r[1][2]*z) * div;

		data+=3;
		dst++;
	}
}
#endif
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

		const float div=1.f/( r[2][0]*x + r[2][1]*y + r[2][2]*z);

		dst->x= (r[0][0]*x + r[0][1]*y + r[0][2]*z) * div;
		dst->y= (r[1][0]*x + r[1][1]*y + r[1][2]*z) * div;

		data+=3;
		dst++;
	}
}


void projectPointsSimple(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest)
{
#ifdef CV_SSE4_1
	myProjectPoint_SSE(xyz, R, t, K, dest);//SSE implimentation
#else
	myProjectPoint_BF(xyz, R, t, K, dest);//normal implementation
#endif
}

void projectPointSimple(Point3d& xyz, const Mat& R, const Mat& t, const Mat& K, Point2d& dest)
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

	const float x = (float)xyz.x+tt[0];
	const float y = (float)xyz.y+tt[1];
	const float z = (float)xyz.z+tt[2];

	const float div=1.f/( r[2][0]*x + r[2][1]*y + r[2][2]*z);
	dest.x= (r[0][0]*x + r[0][1]*y + r[0][2]*z) * div;
	dest.y= (r[1][0]*x + r[1][1]*y + r[1][2]*z) * div;
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

void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub, vector<Point2f>& pt, Mat& depth)
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
		//like cv::projectPoints
		projectPointsSimple(xyz,R,t,K,pt);
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

void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub)
{
	vector<Point2f> pt(image.size().area());
	Mat depth = 10000.f*Mat::ones(image.size(),CV_32F);

	projectImagefromXYZ(image, destimage, xyz, R, t, K, dist, mask, isSub,  pt, depth);
}

template <class T>
static void reprojectXYZ_(const Mat& depth, Mat& xyz, double f)
{
	if(xyz.empty())xyz=Mat::zeros(depth.size().area(),1,CV_32FC3);

	const float bigZ = 10000.f;
	const float fxinv = (float)(1.0/f);
	const float fyinv = (float)(1.0/f);
	const float cw = (depth.size().width-1)*0.5f;
	const float ch = (depth.size().height-1)*0.5f;

	T* dep = (T*)depth.ptr<T>(0);
	float* data=xyz.ptr<float>(0);
	//#pragma omp parallel for
	for(int j=0;j<depth.rows;j++)
	{
		float b = j-ch;
		const float y = b*fyinv;

		float x = (-cw)*fxinv;
		for(int i=0;i<depth.cols;i++)
		{
			float z = (T)*dep;
			data[0]=x*z;
			data[1]=y*z;
			data[2]= (z==0) ?bigZ:z;

			data+=3,dep++;
			x+=fxinv;
		}
	}
}

void reprojectXYZ(const Mat& depth, Mat& xyz, double f)
{
	if(depth.type()==CV_8U)
	{
		reprojectXYZ_<uchar>(depth, xyz, f);
	}
	else if(depth.type()==CV_16S)
	{
		reprojectXYZ_<short>(depth, xyz, f);
	}
	else if(depth.type()==CV_16U)
	{
		reprojectXYZ_<unsigned short>(depth, xyz, f);
	}
	else if(depth.type()==CV_32F)
	{
		reprojectXYZ_<float>(depth, xyz, f);
	}
}

void reprojectXYZ(const Mat& depth, Mat& xyz, Mat& intrinsic, Mat& distortion, float a, float b)
{
	if(xyz.empty())xyz=Mat::zeros(depth.size().area(),1,CV_32FC3);

	const float bigZ = 10000.f;
	const float fxinv = (float)(1.0/intrinsic.at<double>(0,0));
	const float fyinv = (float)(1.0/intrinsic.at<double>(1,1));
	const float cw = (float)intrinsic.at<double>(0,2);
	const float ch = (float)intrinsic.at<double>(1,2);
	const float k0 = (float)distortion.at<double>(0,0);
	const float k1 = (float)distortion.at<double>(1,0);
	//#pragma omp parallel for
	for(int j=0;j<depth.rows;j++)
	{
		const float y = (j-ch)*fyinv;
		const float yy=y*y;
		unsigned short* dep = (unsigned short*)depth.ptr<unsigned short>(j);
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

Point3d get3DPointfromXYZ(Mat& xyz, Size& imsize, Point& pt)
{
	Point3d ret;
	ret.x = xyz.at<float>(imsize.width*3*pt.y + 3*pt.x + 0);
	ret.y = xyz.at<float>(imsize.width*3*pt.y + 3*pt.x + 1);
	ret.z = xyz.at<float>(imsize.width*3*pt.y + 3*pt.x + 2);

	return ret;
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

enum
{
	FILL_DISPARITY =0,
	FILL_DEPTH =1
};
void fillOcclusion(Mat& src, int invalidvalue, int disp_or_depth)
{
    if(disp_or_depth==FILL_DEPTH)
    {
        if(src.type()==CV_8U)
        {
            fillOcclusionInv_<uchar>(src, (uchar)invalidvalue,0);
        }
        else if(src.type()==CV_16S)
        {
            fillOcclusionInv_<short>(src, (short)invalidvalue,0);
        }
        else if(src.type()==CV_16U)
        {
            fillOcclusionInv_<unsigned short>(src, (unsigned short)invalidvalue,0);
        }
        else if(src.type()==CV_32F)
        {
            fillOcclusionInv_<float>(src, (float)invalidvalue,0.f);
        }
    }
    else
    {
        if(src.type()==CV_8U)
        {
            fillOcclusion_<uchar>(src, (uchar)invalidvalue,255);
        }
        else if(src.type()==CV_16S)
        {
			fillOcclusion_<short>(src, (short)invalidvalue,SHRT_MAX);
        }
        else if(src.type()==CV_16U)
        {
			fillOcclusion_<unsigned short>(src, (unsigned short)invalidvalue,USHRT_MAX);
        }
        else if(src.type()==CV_32F)
        {
			fillOcclusion_<float>(src, (float)invalidvalue,FLT_MAX);
        }
    }
}

void disp16S2depth16U(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
{
	if(dest.empty())dest = Mat::zeros(src.size(),CV_16U);
	if(dest.type()!=CV_16U)dest = Mat::zeros(src.size(),CV_16U);

	
#if CV_SSE4_1
	const int ssesize = src.size().area()/16;
	const int remsize = src.size().area()-16*ssesize;
	short* s=src.ptr<short>(0);
	ushort*  d=dest.ptr<ushort>(0);
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

			_mm_stream_si128((__m128i*)(d),_mm_packs_epi32(_mm_cvtps_epi32(v1),_mm_cvtps_epi32(v2)));
			_mm_stream_si128((__m128i*)(d+8),_mm_packs_epi32(_mm_cvtps_epi32(v3),_mm_cvtps_epi32(v4)));
				
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

			_mm_stream_si128((__m128i*)(d),_mm_packs_epi32(_mm_cvtps_epi32(v1),_mm_cvtps_epi32(v2)));
			_mm_stream_si128((__m128i*)(d+8),_mm_packs_epi32(_mm_cvtps_epi32(v3),_mm_cvtps_epi32(v4)));

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
	
#else
	Mat temp;
	divide(a*focal_baseline,src,temp);
	add(temp,b,temp);
	temp.convertTo(dest,CV_8U);
#endif
}


void depth32F2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
{
	if(dest.empty())dest = Mat::zeros(src.size(),CV_8U);
	if(dest.type()!=CV_8U)dest = Mat::zeros(src.size(),CV_8U);

#if CV_SSE4_1
	const int ssesize = src.size().area()/16;
	const int remsize = src.size().area()-16*ssesize;
	float* s=src.ptr<float>(0);
	uchar*  d=dest.ptr<uchar>(0);
	const __m128 maf = _mm_set1_ps(a*focal_baseline);
	if(b==0.f)
	{
		for(int i=0;i<ssesize;i++)
		{
			__m128 v1 = _mm_load_ps(s);
			__m128 v2 = _mm_load_ps(s+4);
			__m128 v3 = _mm_load_ps(s+8);
			__m128 v4 = _mm_load_ps(s+12);

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

			__m128 v1 = _mm_load_ps(s);
			__m128 v2 = _mm_load_ps(s+4);
			__m128 v3 = _mm_load_ps(s+8);
			__m128 v4 = _mm_load_ps(s+12);

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
	
#else
	Mat temp;
	divide(a*focal_baseline,src,temp);
	add(temp,b,temp);
	temp.convertTo(dest,CV_8U);
#endif
}

void depth16U2disp8U(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
{
	if(dest.empty())dest = Mat::zeros(src.size(),CV_8U);
	if(dest.type()!=CV_8U)dest = Mat::zeros(src.size(),CV_8U);

#if CV_SSE4_1
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
	
#else
	Mat temp;
	divide(a*focal_baseline,src,temp);
	add(temp,b,temp);
	temp.convertTo(dest,CV_8U);
#endif
}

void disp8U2depth32F(Mat& src, Mat& dest, const float focal_baseline, float a, float b)
{
	if(dest.empty())dest = Mat::zeros(src.size(),CV_32F);
	if(dest.type()!=CV_32F)dest = Mat::zeros(src.size(),CV_32F);

#if CV_SSE4_1
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
		*d = a*focal_baseline / *s + b;
		s++;
		d++;
	}
#else
	src.convertTo(dest,CV_32F);
	divide(a*focal_baseline,dest,dest);
	add(dest,b,dest);
#endif
}

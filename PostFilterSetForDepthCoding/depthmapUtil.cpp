#include "config.h"
#include "filter.h"

#if CV_SSE4_1
#include <smmintrin.h>
#endif

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

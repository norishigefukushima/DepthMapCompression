#include "filter.h"
#include "config.h"
#if CV_SSE4_1
#include <smmintrin.h>
#endif

template<class T>
void blurRemoveMinMaxBase_(Mat& src, Mat& dest, const int r)
{
	const Size ksize = Size(2*r+1,2*r+1);
	if(src.data!=dest.data)src.copyTo(dest);

	Mat xv;
	Mat nv;
	Mat element = Mat::ones(2*r+1,2*r+1,CV_8U);
	dilate(src,xv,element);
	erode(src,nv,element);

	Mat mind;
	Mat maxd;
	Mat mask;
	absdiff(src,nv,mind);//can move to loop
	absdiff(src,xv,maxd);//
	min(mind,maxd,mask);//

	T* n = nv.ptr<T>(0);
	T* x = xv.ptr<T>(0);
	T* d = dest.ptr<T>(0);
	T* nd = mind.ptr<T>(0);
	T* mk = mask.ptr<T>(0);

	const int remsize = src.size().area();
	for(int i=0;i<remsize;i++)
	{
		{
			if(nd[i]==mk[i])
			{
				d[i]=n[i];
			}
			else
			{
				d[i]=x[i];
			}
		}
	}
}

template<class T>
void blurRemoveMinMax_(Mat& src, Mat& dest, const int r)
{
	const Size ksize = Size(2*r+1,2*r+1);
	if(src.data!=dest.data)src.copyTo(dest);

	Mat xv;
	Mat nv;
	Mat element = Mat::ones(2*r+1,2*r+1,CV_8U);
	dilate(src,xv,element);
	erode(src,nv,element);

	Mat mind;
	Mat maxd;
	Mat mask;
	absdiff(src,nv,mind);//can move to loop
	absdiff(src,xv,maxd);//
	min(mind,maxd,mask);//

	T* n = nv.ptr<T>(0);
	T* x = xv.ptr<T>(0);
	T* d = dest.ptr<T>(0);
	T* nd = mind.ptr<T>(0);
	T* mk = mask.ptr<T>(0);

	int remsize = src.size().area();

#if CV_SSE4_1
	if(src.depth()==CV_8U)
	{

		const int ssesize = src.size().area()/16;
		remsize = src.size().area()-ssesize*16;	
		for(int i=0;i<ssesize;i++)
		{
			__m128i mmk = _mm_load_si128((__m128i*)mk);
			__m128i mnd = _mm_load_si128((__m128i*)nd);

			__m128i mmn = _mm_load_si128((__m128i*)n);
			__m128i mmx = _mm_load_si128((__m128i*)x);
			__m128i msk =  _mm_cmpeq_epi8(mnd,mmk);
			_mm_store_si128((__m128i*)d,_mm_blendv_epi8(mmx,mmn,msk));
			nd+=16;
			mk+=16;
			d+=16;
			n+=16;
			x+=16;
		}
	}
	else if(src.depth()==CV_16S || src.depth()==CV_16U)
	{

		const int ssesize = src.size().area()/8;
		remsize = src.size().area()-ssesize*8;	
		for(int i=0;i<ssesize;i++)
		{
			__m128i mmk = _mm_load_si128((__m128i*)mk);
			__m128i mnd = _mm_load_si128((__m128i*)nd);

			__m128i mmn = _mm_load_si128((__m128i*)n);
			__m128i mmx = _mm_load_si128((__m128i*)x);
			__m128i msk =  _mm_cmpeq_epi16(mnd,mmk);
			_mm_store_si128((__m128i*)d,_mm_blendv_epi8(mmx,mmn,msk));
			nd+=8;
			mk+=8;
			d+=8;
			n+=8;
			x+=8;
		}
	}
	else if(src.depth()==CV_32F)
	{

		const int ssesize = src.size().area()/4;
		remsize = src.size().area()-ssesize*4;	
		for(int i=0;i<ssesize;i++)
		{
			__m128 mmk = _mm_load_ps((float*)mk);
			__m128 mnd = _mm_load_ps((float*)nd);

			__m128 mmn = _mm_load_ps((float*)n);
			__m128 mmx = _mm_load_ps((float*)x);
			__m128 msk =  _mm_cmpeq_ps(mnd,mmk);
			_mm_store_ps((float*)d,_mm_blendv_ps(mmx,mmn,msk));
			nd+=4;
			mk+=4;
			d+=4;
			n+=4;
			x+=4;
		}
	}
	else if(src.depth()==CV_64F)
	{

		const int ssesize = src.size().area()/2;
		remsize = src.size().area()-ssesize*2;	
		for(int i=0;i<ssesize;i++)
		{
			__m128d mmk = _mm_load_pd((double*)mk);
			__m128d mnd = _mm_load_pd((double*)nd);

			__m128d mmn = _mm_load_pd((double*)n);
			__m128d mmx = _mm_load_pd((double*)x);
			__m128d msk =  _mm_cmpeq_pd(mnd,mmk);
			_mm_store_pd((double*)d,_mm_blendv_pd(mmx,mmn,msk));
			nd+=2;
			mk+=2;
			d+=2;
			n+=2;
			x+=2;
		}
	}
#endif
	for(int i=0;i<remsize;i++)
	{
		{
			if(nd[i]==mk[i])
			{
				d[i]=n[i];
			}
			else
			{
				d[i]=x[i];
			}
		}
	}
}

void blurRemoveMinMax(Mat& src, Mat& dest, const int r)
{
	if(src.channels()==1)
	{
		if(src.depth()==CV_8U)
		blurRemoveMinMax_<uchar>(src,dest,r);
		else if(src.depth()==CV_16S)
			blurRemoveMinMax_<short>(src,dest,r);
		else if(src.depth()==CV_16U)
			blurRemoveMinMax_<ushort>(src,dest,r);
		else if(src.depth()==CV_32F)
			blurRemoveMinMax_<float>(src,dest,r);
		else if(src.depth()==CV_64F)
			blurRemoveMinMax_<double>(src,dest,r);
	}
	else
	{
		vector<Mat> v;
		split(src,v);

		if(src.depth()==CV_8U)
		for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMax_<uchar>(v[i],v[i],r);
		else if(src.depth()==CV_16S)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMax_<short>(v[i],v[i],r);
		else if(src.depth()==CV_16U)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMax_<ushort>(v[i],v[i],r);
		else if(src.depth()==CV_32F)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMax_<float>(v[i],v[i],r);
		else if(src.depth()==CV_64F)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMax_<double>(v[i],v[i],r);
		
		merge(v,dest);
	}
}

void blurRemoveMinMaxBase(Mat& src, Mat& dest, const int r)
{
	if(src.channels()==1)
	{
		if(src.depth()==CV_8U)
		blurRemoveMinMaxBase_<uchar>(src,dest,r);
		else if(src.depth()==CV_16S)
			blurRemoveMinMaxBase_<short>(src,dest,r);
		else if(src.depth()==CV_16U)
			blurRemoveMinMaxBase_<ushort>(src,dest,r);
		else if(src.depth()==CV_32F)
			blurRemoveMinMaxBase_<float>(src,dest,r);
		else if(src.depth()==CV_64F)
			blurRemoveMinMaxBase_<double>(src,dest,r);
	}
	else
	{
		vector<Mat> v;
		split(src,v);

		if(src.depth()==CV_8U)
		for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMaxBase_<uchar>(v[i],v[i],r);
		else if(src.depth()==CV_16S)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMaxBase_<short>(v[i],v[i],r);
		else if(src.depth()==CV_16U)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMaxBase_<ushort>(v[i],v[i],r);
		else if(src.depth()==CV_32F)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMaxBase_<float>(v[i],v[i],r);
		else if(src.depth()==CV_64F)
			for(int i=0;i<(int)v.size();i++)
			blurRemoveMinMaxBase_<double>(v[i],v[i],r);
		
		merge(v,dest);
	}
}

template<class T>
static void maxFilter_sp(const Mat& src, Mat& dest,int width, const T maxval,int borderType)
{
	if(src.channels()!=1)return;
	if(width==1){src.copyTo(dest);return;}
	if(dest.empty())dest=Mat::zeros(src.size(),src.type());

	Size size = src.size();

	Mat sim;
	int radiusx = width/2;
	copyMakeBorder( src, sim, 0, 0, radiusx, radiusx, borderType );

	const int st = width - 1;
	//#pragma omp parallel for
	for(int i = 0; i < size.height; i++ )
	{
		const T* sptr = sim.ptr<T>(i);
		T* dptr = dest.ptr<T>(i);

		T prev = maxval;
		for(int k = 0; k < width; k++ )
			prev = max(prev,sptr[+ k]);
		dptr[0] = prev;
		T ed = sptr[0];
		for(int j = 1; j < size.width; j++ )
		{
			if(prev<=sptr[j + st])
			{
				prev = sptr[j + st];
				dptr[j] = prev;	
			}
			else if(ed!=prev)
			{
				dptr[j] = prev;	
				ed = sptr[j];
			}
			else
			{
				T maxv=maxval;
				for(int k = 0; k < width; k++ )
				{
					maxv = max(maxv,sptr[j + k]);
				}
				dptr[j] = maxv;
				prev = maxv;
				ed = sptr[j];
			}
		}		
	}
}
template<class T>
static void maxFilter_(const Mat& src, Mat& dest, Size ksize, T maxval,int borderType)
{
	maxFilter_sp<T>(src,dest,ksize.width,maxval,borderType);
	Mat temp = dest.t();
	Mat temp2;
	maxFilter_sp<T>(temp,temp2,ksize.height,maxval,borderType);
	Mat(temp2.t()).copyTo(dest);
}
void maxFilter(const Mat& src, Mat& dest, Size ksize, int borderType)
{
	if(src.type()==CV_8U)
	{
		maxFilter_<uchar>(src,dest,ksize,0,borderType);
	}
	if(src.type()==CV_16S)
	{
		maxFilter_<short>(src,dest,ksize,SHRT_MIN,borderType);
	}
	if(src.type()==CV_16U)
	{
		maxFilter_<ushort>(src,dest,ksize,0,borderType);
	}
	if(src.type()==CV_32F)
	{
		maxFilter_<float>(src,dest,ksize,FLT_MIN,borderType);
	}
}

template<class T>
static void minFilter_sp(const Mat& src, Mat& dest,int width, const T maxval,int borderType)
{
	if(src.channels()!=1)return;
	if(width==1){src.copyTo(dest);return;}
	if(dest.empty())dest=Mat::zeros(src.size(),src.type());

	Size size = src.size();

	Mat sim;
	int radiusx = width/2;
	copyMakeBorder( src, sim, 0, 0, radiusx, radiusx, borderType );

	const int st = width - 1;
	//#pragma omp parallel for
	for(int i = 0; i < size.height; i++ )
	{
		const T* sptr = sim.ptr<T>(i);
		T* dptr = dest.ptr<T>(i);

		T prev = maxval;
		for(int k = 0; k < width; k++ )
			prev = min(prev,sptr[+ k]);
		dptr[0] = prev;
		T ed = sptr[0];
		for(int j = 1; j < size.width; j++ )
		{
			if(prev>=sptr[j + st])
			{
				prev = sptr[j + st];
				dptr[j] = prev;	
			}
			else if(ed!=prev)
			{
				dptr[j] = prev;	
				ed = sptr[j];
			}
			else
			{
				T maxv=maxval;
				for(int k = 0; k < width; k++ )
				{
					maxv = min(maxv,sptr[j + k]);
				}
				dptr[j] = maxv;
				prev = maxv;
				ed = sptr[j];
			}
		}		
	}
}
template<class T>
static void minFilter_(const Mat& src, Mat& dest, Size ksize, T maxval,int borderType)
{
	minFilter_sp<T>(src,dest,ksize.width,maxval,borderType);
	Mat temp = dest.t();
	Mat temp2;
	minFilter_sp<T>(temp,temp2,ksize.height,maxval,borderType);
	Mat(temp2.t()).copyTo(dest);
}
void minFilter(const Mat& src, Mat& dest, Size ksize, int borderType)
{
	if(src.type()==CV_8U)
	{
		minFilter_<uchar>(src,dest,ksize,255,borderType);
	}
	if(src.type()==CV_16S)
	{
		minFilter_<short>(src,dest,ksize,SHRT_MAX,borderType);
	}
	if(src.type()==CV_16U)
	{
		minFilter_<ushort>(src,dest,ksize,USHRT_MAX,borderType);
	}
	if(src.type()==CV_32F)
	{
		minFilter_<float>(src,dest,ksize,FLT_MAX,borderType);
	}
}
#include <opencv2/opencv.hpp>
#include "config.h"
#include "util.h"
#include "filter.h"
using namespace cv;
using namespace std;

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

// disp = a * (focal_baseline/depth ) + b;
#define FOCUS 75.0
#define BASELINE 575.0
#define AMP_DISP 2.6

//point cloud rendering
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

static void projectImagefromXYZ(const Mat& image, Mat& destimage, const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, const Mat& dist, Mat& mask, const bool isSub)
{
	vector<Point2f> pt(image.size().area());
	Mat depth = 10000.f*Mat::ones(image.size(),CV_32F);

	projectImagefromXYZ(image, destimage, xyz, R, t, K, dist, mask, isSub,  pt, depth);
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

		if(isFillOcclusion)fillOcclusion(depth,0);
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
		if(isFillOcclusion)fillOcclusion(depth,0);
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
		const float cw = (depth.size().width-1)*0.5f;
		const float ch = (depth.size().height-1)*0.5f;

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
		const float k0 = (float)distortion.at<double>(0,0);
		const float k1 = (float)distortion.at<double>(1,0);
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

		in.push_back(Point2f((float)u,(float)v));
		vector<Point2f> out;

		undistortPoints(in,out,intrinsic,distortion,Mat(),intrinsic);
		//undistortPoints(in,out,Mat::eye(3,3,CV_64F),Mat());

		const double fxinv = 1.0/intrinsic.at<double>(0,0);
		const double fyinv = 1.0/intrinsic.at<double>(1,1);

		//cout<<"(fx,fy): "<<fxinv<<","<<fyinv<<endl;
		const float dcx = out[0].x-(float)intrinsic.at<double>(0,2);
		const float dcy = out[0].y-(float)intrinsic.at<double>(1,2);

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
		Point v = Point(cvRound(r*cos(i/20.0)+320-r),cvRound(r*sin(i/20.0)+240));

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

			fillOcclusion(srcDepth,0,FILL_DISPARITY);
			Mat tr=srcDepth.t();
			fillOcclusion(tr,0,FILL_DISPARITY);
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
			fillOcclusion(srcDepth16,0);
			Mat tr=srcDepth16.t();
			fillOcclusion(tr,0);
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
			
			fillOcclusion(srcDepth,0,FILL_DISPARITY);
			Mat tr=srcDepth.t();
			fillOcclusion(tr,0,FILL_DISPARITY);
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
#if JPEG_TURBO
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
				pfs.filterDisp8U2Depth16U(disp,kd.depth,FOCUS,BASELINE,AMP_DISP,mr,gr,br,dr,(float)thresh,FULL_KERNEL);
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
			projectPointSimple(look,R,t,k,ptf);
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
	if(!checkHardwareSupport(CV_CPU_SSE4_1))cout<<"The CPU is not support SSE4.1\n";
	//Mat src = imread("dataset/kinect/desk_1_1.png");
	//Mat depth16 = imread("dataset/kinect/desk_1_1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
	
	Mat src = imread("dataset/kinect/meeting_small_1_1.png");
	Mat depth16 = imread("dataset/kinect/meeting_small_1_1_depth.png",CV_LOAD_IMAGE_UNCHANGED);

	projectTest(src,depth16);
}
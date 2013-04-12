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

static void onMouse(int event, int x, int y, int flags, void* param)
{
	Point* ret=(Point*)param;

	if(flags == CV_EVENT_FLAG_LBUTTON)
	{
		ret->x=x;
		ret->y=y;
	}
}


void pointcloudTest(Mat& image, Mat& srcDepth16)
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

	int px = image.cols/2;
	createTrackbar("look at x","image",&px,image.cols-1);
	int py = image.rows/2;
	createTrackbar("look at y","image",&py,image.rows-1);
	int sub=2;
	createTrackbar("render Opt","image",&sub,3);
	int sw = 0;
	createTrackbar("sw","image",&sw,2);

	Mat destImage(image.size(),CV_8UC3);	//rendered image
	Mat view;//drawed image

	float focal_length=510.f;
	Mat k = Mat::eye(3,3,CV_64F)*focal_length;
	k.at<double>(0,2)=(image.cols-1)*0.5;
	k.at<double>(1,2)=(image.rows-1)*0.5;
	k.at<double>(2,2)=1.0;

	fillOcclusion(srcDepth16,0);
	Mat tr=srcDepth16.t();
	fillOcclusion(tr,0);
	transpose(tr,srcDepth16);
	Mat xyz;
	reprojectXYZ(srcDepth16,xyz,focal_length);

	Point3d srcview;
	Point3d look = get3DPointfromXYZ(xyz,image.size(),pt);

	int count=0;
	bool isDrawLine = true;
	bool isWrite=false;
	bool isLookat = false;


	int postFilterMethod = 2;
	PostFilterSet pfs;

	CalcTime tm("total");
	int key=0;
	Mat srcDisp;
	Mat depthF;

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
			depth16U2disp8U(srcDepth16, srcDisp, FOCUS*BASELINE, 2.6f);

			fillOcclusion(srcDisp,0,FILL_DISPARITY);
			Mat tr=srcDisp.t();
			fillOcclusion(tr,0,FILL_DISPARITY);
			transpose(tr,srcDisp);

			if(isH264)
			{
				double bpp;
				int size;
				degradeImagex264(srcDisp, disp, 0, size, bpp);
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
				imencode(".png",srcDisp,buff,param);
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

			depth16U2disp8U(srcDepth16, srcDisp, FOCUS*BASELINE, 2.6f);
			imencode(".png",srcDepth16,buff,param);
			srcDisp.copyTo(disp);
			bps = 30*8.0*buff.size()/1000000.0;

		}
		else//8 but JPEG
		{	
			depth16U2disp8U(srcDepth16, srcDisp, FOCUS*BASELINE, 2.6f);

			fillOcclusion(srcDisp,0,FILL_DISPARITY);
			Mat tr=srcDisp.t();
			fillOcclusion(tr,0,FILL_DISPARITY);
			transpose(tr,srcDisp);

			if(isH264)
			{
				int qp = (50- q/2) + 1;
				double bpp;
				int size;
				degradeImagex264(srcDisp, disp, qp, size, bpp);
				bps = 30*8.0*size/1000000.0;
			}
			else
			{
#if JPEG_TURBO
				int size;
				double bpp;
				degradeJPEG(srcDisp,disp,q,0,true,size,bpp);
				bps = 30*8.0*size/1000000.0;
#else			
				vector<uchar> buff;
				vector<int> param(2);
				param[0] = CV_IMWRITE_JPEG_QUALITY;
				param[1] = q;
				imencode(".jpg",srcDisp,buff,param);
				disp = imdecode(buff,0);
				bps = 30*8.0*buff.size()/1000000.0;
#endif

			}
		}

		//post filter set for coded depth map
		Mat dshow;
		if(q==102)//case raw
		{
			srcDepth16.convertTo(depthF,CV_32F);
			depth32F2disp8U(depthF, dshow, FOCUS*BASELINE, 2.6f,0.f);
		}
		else// case compression
		{
			//CalcTime t("post filter");
			if(postFilterMethod==2)
			{
				pfs.filterDisp8U2Depth32F(disp,depthF,FOCUS,BASELINE,AMP_DISP,mr,gr,br,dr,(float)thresh,FULL_KERNEL);
				depth32F2disp8U(depthF, dshow, FOCUS*BASELINE, 2.6f,0.f);
			}
			else if(postFilterMethod==1)
			{
				boundaryReconstructionFilter(disp,disp,Size(13,13),1.0,1.0,1.0);
				disp8U2depth32F(disp,depthF,FOCUS*BASELINE,AMP_DISP,0.f);
				depth32F2disp8U(depthF, dshow, FOCUS*BASELINE, 2.6f,0.f);
			}
			else
			{
				disp8U2depth32F(disp,depthF,FOCUS*BASELINE,AMP_DISP,0.f);
				depth32F2disp8U(depthF, dshow, FOCUS*BASELINE, 2.6f,0.f);
			}
		}

		{
			//CalcTime t(" depth2xyz projection");
			reprojectXYZ(depthF,xyz,focal_length);
		}
		
		Mat R = Mat::eye(3,3,CV_64F);
		Mat t = Mat::zeros(3,1,CV_64F);
		t.at<double>(0,0)=x-initx;
		t.at<double>(1,0)=y-inity;
		t.at<double>(2,0)=-z+initz;

		srcview = Point3d(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));
		if(isLookat)
		{	
			look = get3DPointfromXYZ(xyz,image.size(),Point(px,py));
		}
		lookat(look,srcview,R);
		Mat r;
		eular2rot(pitch-90.0,0.0,yaw - 90,r);
		R = r*R;

		//project 3D point image
		if(sw==0)//image view
		{
			if(sub>0)
				projectImagefromXYZ(image,destImage,xyz,R,t,k,Mat(),Mat(),true);
			else
				projectImagefromXYZ(image,destImage,xyz,R,t,k,Mat(),Mat(),false);
		}
		else//depth map view
		{

			Mat dispC;
			if(sw==1)
				cvtColor(dshow,dispC,CV_GRAY2BGR);
			else
				cv::applyColorMap(dshow,dispC,2);

			if(sub>0)
				projectImagefromXYZ(dispC,destImage,xyz,R,t,k,Mat(),Mat(),true);
			else
				projectImagefromXYZ(dispC,destImage,xyz,R,t,k,Mat(),Mat(),false);
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

void simpleTest(Mat& depth16)//input is 16 bit unsinged short depth map
{
	//convert 16 bit depth map to 8 bit disparity map
	Mat disp8;
	depth16U2disp8U(depth16,disp8,FOCUS*BASELINE,AMP_DISP);//AMP_DISP is multiplyer of disparity map for visibility. disparity = FOCUS*BASELINE/depth * AMP_DISP
	fillOcclusion(disp8,0,FILL_DISPARITY); //fill occlusion by using nearest disparity values

	//encode disparity map using opencv functions
	vector<uchar> buff;//buffer for coded disparity map
	vector<int> param(2);//parameter for JPEG coding. 
	param[0] = CV_IMWRITE_JPEG_QUALITY;
	param[1] = 50;//The quality fuctor is 50. (0:lowest, 100:highest)
	imencode(".jpg",disp8,buff,param);//encode disparity map
	cout<<format("compressed size/raw size: %d byte / %d byte = %f",buff.size(),depth16.size().area()*2,buff.size()/(double)(depth16.size().area()*2))<<endl;
	Mat disp8coded = imdecode(buff,0);//decode disparity map

	//perform post filter set to remove coding distortion
	Mat disp8filtered;//post filtered disparity map
	PostFilterSet pfs;//class of our post filter set
	pfs(disp8coded,disp8filtered,2,1,3,5,10);//post filter set

	//visualize //colored disparity map
	Mat disp8C, disp8codedC, disp8filteredC;
	applyColorMap(disp8, disp8C,2);
	applyColorMap(disp8coded, disp8codedC,2);
	applyColorMap(disp8filtered, disp8filteredC,2);
	imshow("input",disp8C);
	imshow("coded",disp8codedC);
	imshow("filtered",disp8filteredC);

	//comapre coded and filtered one, 'q' key is quit function.
	guiAlphaBlend(disp8codedC,disp8filteredC);
}

int main(int argc, char** argv)
{
	if(!checkHardwareSupport(CV_CPU_SSE4_1))cout<<"The CPU is not support SSE4.1\nPlease comment out CV_SSE4_ 1 in config.h\n";
	cout<<"quit key is 'q'\n\n";
	//read image
	//Mat src = imread("dataset/kinect/desk_1_1.png");
	//Mat depth16 = imread("dataset/kinect/desk_1_1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
	Mat src = imread("dataset/kinect/meeting_small_1_1.png");
	Mat depth16 = imread("dataset/kinect/meeting_small_1_1_depth.png",CV_LOAD_IMAGE_UNCHANGED);

	//(1) The simplest example of our post filter set
	//simpleTest(depth16);

	//(2) App for depth map compression
	pointcloudTest(src,depth16);
}
#include "util.h"
using namespace std;


static void writeYUVGray(string fname, Mat& src)
{
	Size size =src.size();
	FILE* fp = fopen(fname.c_str(),"wb");
	if(fp==NULL)cout<<fname<<" open error\n";
	const int fsize = size.area() + size.area()*2/4;

	//fseek(fp,fsize*frame,SEEK_END);
	uchar* buff = new uchar[fsize];
	memset(buff,0,fsize);
	memcpy(buff,src.data,size.area());
	//for(int i=0;i<100;i++)
		fwrite(buff,sizeof(char),fsize,fp);
	fclose(fp);
	delete[] buff;
}

static void readYUVGray(string fname, Mat& dest, Size size, int frame)
{
	dest.create(size,CV_8U);
	FILE* fp = fopen(fname.c_str(),"rb");
	if(fp==NULL)cout<<fname<<" open error\n";
	const int fsize = size.area() + size.area()*2/4;

	//fseek(fp,fsize*frame,SEEK_END);
	
	fread(dest.data,sizeof(char),size.area(),fp);
	//cout<<size.area()<<endl;
	fclose(fp);
	//imshow("aa",dest);waitKey();
}
double degradeImagex264(Mat& src, Mat& dest, int qp, int& size, double& bpp)
{
	const int headersize=6310;

	FILE* x264;
	char cmd[1024];

	writeYUVGray("out.yuv",src);
	qp = max(min(51,qp),0);
	
	//4x4 DCT
	sprintf(cmd,"ffmpeg -y -s %dx%d -i out.yuv -vcodec libx264 -cqp %d -coder 1 -trellis 2 -flags -loop -preset veryslow out.avi",src.cols,src.rows,qp);

	//8x8 DCT
	//sprintf(cmd,"ffmpeg -y -s %dx%d -i out.yuv -vcodec libx264 -cqp %d -coder 1 -trellis 2 -flags -loop -flags2 +dct8x8 -preset veryslow out.avi",src.cols,src.rows,qp);

	cout<<cmd<<endl;
	x264 = _popen(cmd,"w");
	fflush(x264);_pclose(x264);

	sprintf(cmd,"ffmpeg -y -i out.avi depth.yuv");
	x264 = _popen(cmd,"w");
	fflush(x264);_pclose(x264);

	FILE* fp = fopen("out.avi","rb");
	fpos_t s = 0;
	/* ファイルサイズを調査 */ 
	fseek(fp,0,SEEK_END); 
	fgetpos(fp,&s); 
	fclose(fp);
	size =(int)s;
	size-=headersize;
	bpp = 8.0*size/(double)(src.cols*src.rows);
	
	readYUVGray("depth.yuv",dest,src.size(),0);
	
	return bpp;
}


void writeYUV(Mat& src_, string name, int mode=1)
{
	Mat src;cvtColor(src_,src,CV_BGR2YUV);
	
	int s = 1;
	if(src.type()==CV_16S) s=2;
	if(mode==0 || s==1)
	{
		FILE* fp = fopen(name.c_str(),"wb");
		fwrite(src.data,sizeof(uchar),src.size().area()*s,fp);
		fclose(fp);
	}
	else
	{

		uchar* buff = new uchar[sizeof(uchar),src.size().area()*2];
		int size = 0;
		int s2=0;
		short* s=src.ptr<short>(0);
		for(int i=0;i<src.size().area();i++)
		{
			int v = s[i]+128;
			if(v <255 && v>=0)
			{
				buff[size++]= v;

			}
			else
			{
				s2++;
				buff[size++]=255;
				short* a = (short*)buff;
				*a = v-128;
				buff+=2;
				size+=2;
			}
		}
		FILE* fp = fopen(name.c_str(),"wb");
		fwrite(buff,sizeof(uchar),size,fp);
		fclose(fp);
		delete[] buff;
	}
}

double degradeImagex2642(Mat& src, Mat& dest, int qp, int& size, double& bpp)
{
	int headersize;
	if(qp==0)
		headersize=6540;
	else
		headersize=6312;

	FILE* x264;
	char cmd[1024];

	writeYUVGray("out.yuv",src);
	qp = max(min(51,qp),0);
	//sprintf(cmd,"ffmpeg -y -i depthx264.png -vcodec libx264 -tune stillimage -cqp %d -flags -loop out.avi",qp+3);
	//sprintf(cmd,"ffmpeg -y -i depthx264.png -vcodec libx264 -tune stillimage -cqp %d -coder 1 -flags -loop -flags2 +dct8x8 out.avi",qp+3);
	//sprintf(cmd,"ffmpeg -y -i depthx264.png -vcodec libx264 -tune stillimage -cqp %d -coder 1 -flags2 +dct8x8 out.avi",qp);

	//sprintf(cmd,"ffmpeg -y -s 640x480 -i out.yuv -vcodec libx264 -cqp %d -coder 1 -trellis 2 -flags -loop -flags2 +dct8x8 -psnr out.avi",qp);
	sprintf(cmd,"ffmpeg -y -s %dx%d -i out.yuv -vcodec libx264 -cqp %d -coder 1 -flags -loop -preset veryslow -flags2 +dct8x8 -psnr out.avi",src.cols,src.rows,qp);

	//sprintf(cmd,"ffmpeg -y -s 640x480 -i out.yuv -vcodec libx264 -intra -cqp %d -x264opts 8x8dct=1:aq-mode=2:cabac=1:aq-strength=1.0 out.avi",qp);
	

	//sprintf(cmd,"ffmpeg -y -s 640x480 -i out.yuv -vcodec ffv1 out.avi");

	//sprintf(cmd,"ffmpeg -y -i depthx264.png -vcodec libx264 -tune stillimage -cqp %d -coder 1 out.avi",qp);

	//sprintf(cmd,"ffmpeg -y -i depthx264.png -vcodec libx264 -tune stillimage -cqp %d -coder 1 -flags2 +dct8x8 out.avi",qp);

	//sprintf(cmd,"ffmpeg -y -i depthx264.png -vcodec libx264 -tune stillimage -cqp %d -coder 1 out.avi",qp+3);
	//sprintf(cmd,"ffmpeg -y -i depthx264.png -vcodec libx264 -cqp %d -coder 1 out.avi",qp+3);
	//sprintf(cmd,"ffmpeg -y -i depthx264.png -pix_fmt gray -vcodec libx264 -pix_fmt rgb24 -tune stillimage -cqp %d -coder 1 -flags +gray -flags2 +dct8x8 out.avi",qp+3);
	//sprintf(cmd,"ffmpeg -y -i depthx264.png -pix_fmt yuv420p -vcodec libx264 -tune stillimage -cqp %d -coder 1 -flags -loop -flags2 +dct8x8 out.avi",qp+3);
	cout<<cmd<<endl;
	x264 = _popen(cmd,"w");
	fflush(x264);_pclose(x264);

	//sprintf(cmd,"ffmpeg -y -i out.avi depth.tiff");
	sprintf(cmd,"ffmpeg -y -i out.avi depth.yuv");
	x264 = _popen(cmd,"w");
	fflush(x264);_pclose(x264);

	FILE* fp = fopen("out.avi","rb");
	fpos_t s = 0;
	/* ファイルサイズを調査 */ 
	fseek(fp,0,SEEK_END); 
	fgetpos(fp,&s); 
	fclose(fp);
	size =(int)s;
	size-=headersize;
	bpp = 8.0*size/(double)(src.cols*src.rows);

	//Mat temp = imread("depth.tiff",0);temp.copyTo(dest);
	/*cv::VideoCapture vc("out.avi");
	Mat temp;
	vc>>temp;
	cvtColor(temp,dest,CV_BGR2GRAY);*/
	
	readYUVGray("depth.yuv",dest,src.size(),0);
	
	/*Mat temp = imread("depth.pgm");
	vector<Mat> v;split(temp,v);
	v[0].copyTo(dest);*/
	
	return bpp;
}

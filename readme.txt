Project Page
http://nma.web.nitech.ac.jp/fukushima/research/depthmap_postfilter.html

The solution file of PostFilterSetForDepthCoding.sln is Visual Studio 2010 file.
In the main function, there are 2 examples; one is (1) the simplest example of our post filter set,
named simpleTest(), and the other is (2) point cloud rendering app for depth map compression, named pointcloudTest();

The solution requires OpenCV2.4, Intel Threading Building Blocks (TBB), and OpenNI (if necessary).
In addition, the codes use SSE4.1. If your CPU does not support SSE4.1, code will not work. 
In this case, please change the file of in config.h.
The current setup is as follows (TBB on, SSE4.1 on, OpenNI off, libjpegturbo on);

//for TBB parallelization
 #define HAVE_TBB

//for SSE4.1 SIMD optimization
 #define CV_SSE4_1 1

//for OpenNI Capture
//#define CAP_KINECT 1

//for libjpegturbo: faster jpeg encoder
 #define JPEG_TURBO 1

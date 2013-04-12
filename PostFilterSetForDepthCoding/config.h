#ifndef _CONFIG_H____
#define _CONFIG_H____

//for TBB parallelization. If you do not have TBB, please comment out.
#define HAVE_TBB

//for SSE4.1 SIMD optimization. If you do not have SSE4.1 capable CPU, please comment out.
#define CV_SSE4_1 1

//for OpenNI Capture
//comment out -> image inpout
//else        -> OpenNI with OpenCV input 
//#define CAP_KINECT 1

//for libjpegturbo: faster jpeg encoder
#define JPEG_TURBO 1

#endif
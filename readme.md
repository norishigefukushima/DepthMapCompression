Depth map compression App
=============
Project Page:
http://nma.web.nitech.ac.jp/fukushima/research/depthmap_postfilter.html

This code is implimentation of the following paper.

N. Fukushima, T. Inoue, Y. Ishibashi, "Removing Depth Map Coding Distortion by Using Post Filter Set," Proc. IEEE International Conference on Multimedia & Expo (ICME 2013), July 2013

The solution file of PostFilterSetForDepthCoding.sln is a project file of Visual Studio 2010.
In the main function, there are 2 examples; one is (1) the simplest example of our post filter set,
named simpleTest(), and the other is (2) point cloud rendering app for depth map compression, named pointcloudTest();

The solution requires OpenCV2.45(2.45 or newer), and OpenNI (if necessary).
In addition, the codes use SSE4.1. If your CPU supports SSE4.1, code will be accelerated. 

we can change compression lib and input device with setting config.h file 
The current setup is as follows (OpenNI off, libjpegturbo off);

//for OpenNI Capture
//#define CAP_KINECT 1

//for libjpegturbo: faster jpeg encoder
//#define JPEG_TURBO 1

If you can use Kinect and/or JPEG_turbo, please un-comment out.
The attached lib file of jpeg.lib is 64 bit version. If your OS is 32bit version, please do not use JPEG_TURBO option.  


This software is released under the BSD License, see LICENSE.txt.

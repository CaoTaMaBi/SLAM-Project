#ifndef KINECT_INPUT_H
#define KINECT_INPUT_H

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector> 
#include <pthread.h>

#include <XnCppWrapper.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>

using namespace xn;
using namespace cv;
using namespace std;
using namespace pcl;

struct PointCloud_Color 
{  
    float  X;  
    float  Y;  
    float  Z;  
    float  R;  
    float  G;  
    float  B;  
  
    PointCloud_Color(XnPoint3D pos, XnRGB24Pixel color)  
    {  
        X = pos.X;  
        Y = pos.Y;  
        Z = pos.Z;  
        R = (float)color.nRed / 255;  
        G = (float)color.nGreen / 255;  
        B = (float)color.nBlue / 255;  
    } 
    
    PointCloud_Color(XnPoint3D pos, float r, float g, float b)  
    {  
        X = pos.X;  
        Y = pos.Y;  
        Z = pos.Z;  
        R = r;  
        G = g;  
        B = b;  
    } 
}; 

class Kinect_Input
{
	public:
		XnStatus eResult;
		Context mContext;
		DepthMetaData mDepthMetaData;
		DepthGenerator mDepthGenerator;
		ImageMetaData mImageMetaData;
		ImageGenerator mImageGenerator;
		XnMapOutputMode mapMode;
		const XnDepthPixel* pDepthMap;
		XnRGB24Pixel* pImageMap;
		
		Mat imgDepth16u;
		Mat imgRGB8u;
		Mat depthshow;
		Mat colordepthshow;
		Mat imageshow;
		Mat maskshow;

		vector<PointCloud_Color> color_cloud;
		XnPoint3D* m_depthdata;
		
		pthread_t thread1;
		
	public:
		Kinect_Input();
		void CheckOpenNIError(XnStatus Result, string sStatus);
		void context_initial();
		void depth_create(Context* initial);
		void image_create(Context* initial);
		void mapmode_setup(XnMapOutputMode* setup);
		void viewpoint_correct(ImageGenerator* image);
		void generatordata_start(Context* data);
		void data_read(Context* data);
		void shut_down(Context* stop);
		void start();
		void meta2xnpoint(DepthMetaData &m2x_DepthMetaData);
				
		Mat depth_gray2rainbow(Mat gray);
		Mat mask_generate(Mat rainbow, Mat color);
		
		static void* pointcloud_thread(void* pvData); 
//		vector<PointCloud_Color> pointcloud_generate(DepthGenerator* depth, DepthMetaData* depthmeta, const XnDepthPixel* depthpixel, const XnRGB24Pixel* imgpixel);
//		PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_generate(DepthGenerator* depth, DepthMetaData* depthmeta, const XnDepthPixel* depthpixel, const XnRGB24Pixel* imgpixel);		
		void printout(vector<PointCloud_Color> printpoint, int num);
		void printout(XnPoint3D* printpoint, int num);
		void printout(PointXYZRGB printpoint);
//		pcl::visualization::PCLVisualizer rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
};
#endif

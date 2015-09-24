#ifndef POINTCLOUDPROCESS_H
#define POINTCLOUDPROCESS_H

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector> 
#include <pthread.h>

#include <Kinect_Input.h>
#include <Detect.h>

#include <XnCppWrapper.h>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl-1.7/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.7/pcl/visualization/cloud_viewer.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace xn;
using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;

class PointCloudProcess
{
	public:
		Kinect_Input pki;
		Detect pdt;
		
		pthread_t ptr_cloudthread;
		
		DepthGenerator* m_depthgen;
		DepthMetaData* m_depthmeta;
		const XnDepthPixel* m_pdepthpix;
		const XnRGB24Pixel* m_pimagepix;
		
		Isometry3d m_parament;
		
		
	public:
		PointCloudProcess();
		void map_generation(Mat tmp_rvec, Mat tmp_tvec,DepthGenerator* tmp_pdepth, DepthMetaData* tmp_pdepthmeta, const XnDepthPixel* tmp_pdepthpixel, const XnRGB24Pixel* tmp_pimgpixel);
		PointCloud<pcl::PointXYZRGB>::Ptr pointcloud_generation(DepthGenerator* pcg_pdepth, DepthMetaData* pcg_pdepthmeta, const XnDepthPixel* pcg_pdepthpixel, const XnRGB24Pixel* pcg_pimgpixel);
		Isometry3d matrix_generation(Mat mg_rvec, Mat mg_tvec);
		
		static void* POINTCLOUD_THREAD(void* pvData);
};
#endif

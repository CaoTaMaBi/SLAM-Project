#include <PointCloudProcess.h>

using namespace xn;
using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;

PointCloud<pcl::PointXYZRGB>::Ptr g_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

PointCloudProcess::PointCloudProcess()
{
	int threadwhether=pthread_create(&ptr_cloudthread,NULL,PointCloudProcess::POINTCLOUD_THREAD,this);
	m_parament = Isometry3d::Identity();
}

void* PointCloudProcess::POINTCLOUD_THREAD(void* pvData)
{
	PointCloudProcess *pcp = (PointCloudProcess*) pvData;
	
	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(g_cloud);
	viewer.setBackgroundColor (255, 255, 255);
	viewer.addPointCloud<pcl::PointXYZRGB> (g_cloud, rgb, "sample cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer.addCoordinateSystem (2.0);
	
	while(1)
	{
		g_cloud = (*pcp).pointcloud_generation((*pcp).m_depthgen, (*pcp).m_depthmeta, (*pcp).m_pdepthpix, (*pcp).m_pimagepix);
		
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(g_cloud);	
		viewer.updatePointCloud(g_cloud, rgb, "sample cloud");
		viewer.spinOnce (10);
		if(viewer.wasStopped ())
		{
			cout << "FUCK" << endl;
			break;
		}

		usleep(100);
	}
}

PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcess::pointcloud_generation(DepthGenerator* pcg_pdepth, DepthMetaData* pcg_pdepthmeta, const XnDepthPixel* pcg_pdepthpixel, const XnRGB24Pixel* pcg_pimgpixel)
{
	int serialy=0;
	int serial=0; 
	int pointnumber = (*pcg_pdepthmeta).FullXRes()*(*pcg_pdepthmeta).FullYRes();
	XnPoint3D* depthpointcloud = new XnPoint3D[pointnumber];
	XnPoint3D* realpointcloud = new XnPoint3D[pointnumber];
	PointXYZRGB pclpoint;
	PointCloud<pcl::PointXYZRGB>::Ptr pclcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	
	for(int j = 0; j < (*pcg_pdepthmeta).FullYRes(); ++j)
	{
		serialy = j * (*pcg_pdepthmeta).FullXRes();

		for(int i = 0; i < (*pcg_pdepthmeta).FullXRes(); ++i)
		{
			serial = serialy + i; 
			depthpointcloud[serial].X = i;
			depthpointcloud[serial].Y = j;
			depthpointcloud[serial].Z = pcg_pdepthpixel[serial];
//			printout(depthpointcloud, serial);
		}
	}
	
	(*pcg_pdepth).ConvertProjectiveToRealWorld(pointnumber, depthpointcloud, realpointcloud);
	
	for(int i = 0; i < pointnumber; i++)
	{
		if(realpointcloud[i].Z == 0)
		{
			continue;
		}
		
		pclpoint.x = realpointcloud[i].X;
		pclpoint.y = realpointcloud[i].Y;
		pclpoint.z = realpointcloud[i].Z;
		pclpoint.r = pcg_pimgpixel[i].nRed;
		pclpoint.g = pcg_pimgpixel[i].nGreen;
		pclpoint.b = pcg_pimgpixel[i].nBlue;
		pclcloud->points.push_back(pclpoint);
		
	}
	
	pclcloud->width = (int) pclcloud->points.size ();
	pclcloud->height = 1;

	delete depthpointcloud;
	delete realpointcloud;

	return(pclcloud);
}

Isometry3d PointCloudProcess::matrix_generation(Mat mg_rvec, Mat mg_tvec)
{
	Mat R;
	Matrix3d r;
	Isometry3d T;
	
	Rodrigues (mg_rvec, R);
	cv2eigen(R, r);
	AngleAxisd angle(r);
	Translation<double,3> trans(mg_tvec.at<double>(0,0), mg_tvec.at<double>(0,1), mg_tvec.at<double>(0,2));
	T = angle;
	T(0,3) = mg_tvec.at<double>(0,0);
	T(1,3) = mg_tvec.at<double>(0,1);
	T(2,3) = mg_tvec.at<double>(0,2);
	
	return (T);
}

void PointCloudProcess::map_generation(Mat tmp_rvec, Mat tmp_tvec,DepthGenerator* tmp_pdepth, DepthMetaData* tmp_pdepthmeta, const XnDepthPixel* tmp_pdepthpixel, const XnRGB24Pixel* tmp_pimgpixel)
{
	m_depthgen = tmp_pdepth;
	m_depthmeta = tmp_pdepthmeta;
	m_pdepthpix = tmp_pdepthpixel;
	m_pimagepix = tmp_pimgpixel;
	
	m_parament = matrix_generation(tmp_rvec, tmp_tvec);
}

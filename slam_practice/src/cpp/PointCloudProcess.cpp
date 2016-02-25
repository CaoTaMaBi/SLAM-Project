#include <PointCloudProcess.h>

using namespace xn;
using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;
using namespace g2o;

extern bool running, state1;
extern g2o::SparseOptimizer m_globalOptimizer;

PointCloud<pcl::PointXYZRGB>::Ptr g_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
PointCloud<pcl::PointXYZRGB>::Ptr g_cloudcurrent (new pcl::PointCloud<pcl::PointXYZRGB>);
PointCloud<pcl::PointXYZRGB>::Ptr g_cloudshow (new pcl::PointCloud<pcl::PointXYZRGB>);
#if 0
PointCloudProcess::PointCloudProcess()
{
	m_flag = 0;
	m_flag4det = 0;
	m_flag2det = 0;
	cout << "fuckfuck" <<endl;
	int threadwhether=pthread_create(&ptr_cloudthread,NULL,PointCloudProcess::POINTCLOUD_THREAD,this);  cout << "fuckfuck" <<endl;
}

void* PointCloudProcess::POINTCLOUD_THREAD(void* pvData)
{
	PointCloudProcess *pcp = (PointCloudProcess*) pvData;
	PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	static VoxelGrid<pcl::PointXYZRGB> voxel;
	double gridsize = 0.02;
	
	voxel.setLeafSize( gridsize, gridsize, gridsize );
	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(g_cloud);
	viewer.setBackgroundColor (255, 255, 255);
	viewer.addPointCloud<pcl::PointXYZRGB> (g_cloud, rgb, "sample cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer.addCoordinateSystem (2.0);
	
	while(1)
	{

		if((*pcp).m_flag == 1)
		{
		//	if(m_inlier.rows < 10)
		//		continue;
			
			g_cloudcurrent = (*pcp).pointcloud_generation((*pcp).m_depthgen, (*pcp).m_depthmeta, (*pcp).m_pdepthpix, (*pcp).m_pimagepix);
			transformPointCloud(*g_cloud, *tmp_cloud, (*pcp).m_matrix.matrix());
			*tmp_cloud = *g_cloudcurrent + *tmp_cloud;

			voxel.setInputCloud(tmp_cloud);
			voxel.filter(*g_cloud);			
			(*pcp).m_flag = 0;
		}			
		
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

PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcess::pointcloud_generation(DepthGenerator* pcg_pdepth, DepthMetaData* pcg_pdepthmeta, const XnDepthPixel* pcg_pdepthpixel, XnRGB24Pixel* pcg_pimgpixel)
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
	m_flag4det = 0;
	return(pclcloud);
}


void PointCloudProcess::map_generation(Isometry3d mg_matrix, Mat tmp_inliers, DepthGenerator* tmp_pdepth, DepthMetaData* tmp_pdepthmeta, const XnDepthPixel* tmp_pdepthpixel, XnRGB24Pixel* tmp_pimgpixel, int mg_detflag)
{
	m_depthgen = tmp_pdepth;
	m_depthmeta = tmp_pdepthmeta;
	m_pdepthpix = tmp_pdepthpixel;
	m_pimagepix = tmp_pimgpixel;
//	m_inlier = tmp_inliers;
//	cout << tmp_rvec <<endl;//detect类match点不够时不进行solvepnp操作，此时rvec为空，导致rodrigous计算报错。

	m_matrix = mg_matrix;
	m_flag4det = mg_detflag;
	
	if(m_flag4det == 1)		
		m_flag = 1;
}
#endif


PointCloudProcess::PointCloudProcess()
{
	m_flag = 0;
	m_flag4det = 0;
	m_flag2det = 0;
	cout << "fuckfuck" <<endl;
	int threadwhether=pthread_create(&ptr_cloudthread,NULL,PointCloudProcess::POINTCLOUD_THREAD,this); 
}

void* PointCloudProcess::POINTCLOUD_THREAD(void* pvData)
{
	vector<PointCloud<pcl::PointXYZRGB>> cloudList;

	PointCloudProcess *pcp = (PointCloudProcess*) pvData;
	PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
	VoxelGrid<pcl::PointXYZRGB> voxel;
	PassThrough<pcl::PointXYZRGB> pass;
	double gridsize = 0.2;
	
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0, 4.0);
	
	voxel.setLeafSize( gridsize, gridsize, gridsize );
	
	
	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(g_cloud);//g_cloud);
	viewer.setBackgroundColor (255, 255, 255);
	viewer.addPointCloud<pcl::PointXYZRGB> (g_cloud, rgb, "sample cloud");

	tmp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	tmp_cloud2 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::visualization::PCLVisualizer viewer1("3D Viewer1");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(tmp_cloud2);//g_cloud);
	viewer1.setBackgroundColor (255, 255, 255);
	viewer1.addPointCloud<pcl::PointXYZRGB> (tmp_cloud2, rgb1, "sample cloud1");
//	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	//viewer.addCoordinateSystem (2.0);

	state1 = true;

	//m_globalOptimizer.initializeOptimization();
	
	while(running)
	{
		if((*pcp).m_flag == 1)
		{
			if((*pcp).m_inlier.rows < 10)
				continue;

			m_globalOptimizer.initializeOptimization();
			m_globalOptimizer.optimize(100);//这个地方最好查一下g2o查节点数量的
			
//			uint16_t kfend= (*pcp).m_depthkf.size()-1;
			for(size_t i = 0; i < (*pcp).m_depthkf.size(); i++)
			{	
			
				VertexSE3* vertex = dynamic_cast<VertexSE3*>(m_globalOptimizer.vertex(i+1));
				Isometry3d pose = Eigen::Isometry3d::Identity();
				pose = vertex -> estimate();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//tmp_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
				//tmp_cloud2 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
				
				g_cloudcurrent->clear();
				g_cloudcurrent = (*pcp).pointcloud_generation((*pcp).m_depthgen, (*pcp).m_depthkf[i], (*pcp).m_colorkf[i]);
				//voxel.setInputCloud(g_cloudcurrent);
				//voxel.filter( *tmp_cloud2);
				//pass.setInputCloud( tmp_cloud2 );
				//pass.filter( *g_cloudcurrent );
				
				tmp_cloud->clear();
				transformPointCloud(*g_cloudcurrent, *tmp_cloud, pose.matrix());
				*tmp_cloud2 = *tmp_cloud;
				//*g_cloudcurrent += *tmp_cloud;
				*g_cloud += *tmp_cloud;
				//*tmp_cloud += *g_cloud;
				//*g_cloud = *g_cloudcurrent;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				
				//cloudList.push_back(*g_cloudcurrent);
				//voxel.setInputCloud(tmp_cloud);e
				//voxel.filter(*t);
				//g_cloud = t;
#if 0
				pass.setInputCloud(tmp_cloud);
				pass.filter(*g_cloudcurrent);
					
				transformPointCloud(*g_cloudcurrent, *tmp_cloud, pose.matrix());
				*g_cloud += *tmp_cloud;
				
				tmp_cloud -> clear();
				g_cloudcurrent -> clear();				
#endif
			}
			
			//voxel.setInputCloud(g_cloud);
			//voxel.filter(*g_cloudshow);
//			g_cloud -> clear();
			g_cloudshow->clear();
			*g_cloudshow = *g_cloud;
			g_cloud -> clear();		
			(*pcp).m_flag = 0;
		
		}
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(tmp_cloud2);//g_cloudshow);	
		viewer1.updatePointCloud(tmp_cloud2, rgb1, "sample cloud1");//(g_cloudshow, rgb, "sample cloud");
		//viewer1.spinOnce (1);

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(g_cloudshow);//g_cloudshow);	
		viewer.updatePointCloud(g_cloudshow, rgb, "sample cloud");//(g_cloudshow, rgb, "sample cloud");
		viewer.spinOnce (1);
		
		if(viewer.wasStopped ())
		{
			cout << "FUCK" << endl;
			break;
		}

		usleep(100);
	}
	
	string name;
	/*
	for(int i=0; i<cloudList.size(); i++)
	{
		//int i = cloudList.size()-1;
		stringstream ss;
		ss << i;
		name = ss.str();
		name += ".pcd";
		std::cout << "save: " << name << endl;
		pcl::io::savePCDFile(name, (cloudList[i]));
	}*/
	pcl::io::savePCDFile("final.pcd", *g_cloudshow);
	state1 = false;
}

bool PointCloudProcess::image2pointcloud(uint32_t in_size, XnPoint3D* in_depthCloud, XnPoint3D* in_realCloud)
{
	double cx = 320;
	double cy = 240;
	double fx = 525;
	double fy = 525;
	double scale = 1000;

	for (uint32_t i = 0; i < in_size; i++)
	{
		//std::cout << "X: " << in_depthCloud[i].X << ", " << in_depthCloud[i].Y << ", " << in_depthCloud[i].Z << std::endl;
		in_realCloud[i].Z = double(in_depthCloud[i].Z / scale);
		in_realCloud[i].X = (in_depthCloud[i].X - cx) * in_realCloud[i].Z / fx;
		in_realCloud[i].Y = (in_depthCloud[i].Y - cy) * in_realCloud[i].Z / fy;
		//std::cout << "TTTTTTTTTTTTTT" << i << ": " << in_depthCloud[i].X << ", " << in_depthCloud[i].Y << ", " << in_depthCloud[i].Z << std::endl;
	}
	return true;
}

PointCloud<pcl::PointXYZRGB>::Ptr PointCloudProcess::pointcloud_generation(DepthGenerator pcg_pdepth, XnPoint3D* pcg_depthcloud, XnRGB24Pixel* pcg_pimgpixel)
{
	int serialy=0;
	int serial=0; 
	int pointnumber = 307200;//(*pcg_pdepthmeta).FullXRes()*(*pcg_pdepthmeta).FullYRes();
	XnPoint3D* depthpointcloud = new XnPoint3D[pointnumber];
	XnPoint3D* realpointcloud = new XnPoint3D[pointnumber];
	PointXYZRGB pclpoint;
	PointCloud<pcl::PointXYZRGB>::Ptr pclcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
#if 0	
	for(int j = 0; j < (*pcg_pdepthmeta).FullYRes(); ++j)
	{
		serialy = j * (*pcg_pdepthmeta).FullXRes();

		for(int i = 0; i < (*pcg_pdepthmeta).FullXRes(); ++i)
		{
			serial = serialy + i; 
			depthpointcloud[serial].X = i;
			depthpointcloud[serial].Y = j;
			depthpointcloud[serial].Z = pcg_pdepthpixel[serial];
		}
	}
#endif	
	//pcg_pdepth.ConvertProjectiveToRealWorld(pointnumber, pcg_depthcloud, realpointcloud);
	image2pointcloud(pointnumber, pcg_depthcloud, realpointcloud);
//cout << "MABIYA" <<endl;
#if 1
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
#endif	
	pclcloud->width = (int) pclcloud->points.size ();
	pclcloud->height = 1;

	delete depthpointcloud;
	delete realpointcloud;
	m_flag4det = 0;

//cout << "CACACACACACA" <<endl;	
	return(pclcloud);
}


void PointCloudProcess::map_generation(Isometry3d mg_matrix, Mat tmp_inliers, DepthGenerator mg_depth, vector<XnPoint3D*> mg_depthkeyframe, vector<XnRGB24Pixel*> mg_colorkeyframe, int mg_detflag)
{
//	m_depthgen = mg_depth;
//	m_depthmeta = tmp_pdepthmeta;
///	m_pdepthpix = tmp_pdepthpixel;
//	m_pimagepix = tmp_pimgpixel;
//	m_inlier = tmp_inliers;
//	cout << tmp_rvec <<endl;//detect类match点不够时不进行solvepnp操作，此时rvec为空，导致rodrigous计算报错。

	m_depthgen = mg_depth;
	m_depthkf = mg_depthkeyframe;
//	cout<<m_depthkf.size() << "qawsed" << endl;
	m_colorkf = mg_colorkeyframe;
	m_matrix = mg_matrix;
	//cout << m_matrix.matrix() << endl;
	m_flag4det = mg_detflag;
	m_inlier = tmp_inliers;
//	m_gOptimizer = mg_globalOptimizer;
	
	if(m_flag4det == 1)		
		m_flag = 1;
		
//	m_flag4det = 0;
}

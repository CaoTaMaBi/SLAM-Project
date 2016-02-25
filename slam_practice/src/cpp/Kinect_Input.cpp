#include "Kinect_Input.h"

using namespace std;
using namespace xn;
using namespace cv;
using namespace pcl;

PointCloud<pcl::PointXYZRGB>::Ptr m_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

Kinect_Input::Kinect_Input()
{
	//cout <<"cacacacacacaaca" <<endl;
	eResult = XN_STATUS_OK;
	imgDepth16u.create(480,640,CV_16UC1);
	imgRGB8u.create(480,640,CV_8UC3);
	depthshow.create(480,640,CV_8UC1);
	colordepthshow.create(480,640,CV_8UC3);
	imageshow.create(480,640,CV_8UC3);
	maskshow.create(480,640,CV_8UC3);
	namedWindow("Image");
	namedWindow("Depth");
	namedWindow("ColorDepth");
	namedWindow("Mask");
	m_depthdata = new XnPoint3D[307200];
	
}

void Kinect_Input::CheckOpenNIError(XnStatus Result, string sStatus)
{
	if( Result != XN_STATUS_OK )
	cerr << sStatus << " Error : " << xnGetStatusString( Result ) << endl;
}

void Kinect_Input::context_initial()
{
	eResult = mContext.Init();
	CheckOpenNIError(eResult, "Initialize context");
}

void Kinect_Input::depth_create(Context* initial)
{
	eResult = mDepthGenerator.Create(mContext);
	CheckOpenNIError(eResult, "Create depth generator");
}

void Kinect_Input::image_create(Context* initial)
{
	eResult = mImageGenerator.Create(mContext);
	CheckOpenNIError(eResult, "Creare image generator");
}

void Kinect_Input::mapmode_setup(XnMapOutputMode* setup)
{
	(*setup).nXRes = 640;
	(*setup).nYRes = 480;
	(*setup).nFPS = 30;
	eResult = mDepthGenerator.SetMapOutputMode(*setup);
	eResult = mImageGenerator.SetMapOutputMode(*setup);
}

void Kinect_Input::viewpoint_correct(ImageGenerator* image)
{
	mDepthGenerator.GetAlternativeViewPointCap().SetViewPoint(mImageGenerator);
	
}

void Kinect_Input::generatordata_start(Context* data)
{
	eResult = (*data).StartGeneratingAll();
}

void Kinect_Input::data_read(Context* data)
{
	//delete m_depthdata;
	m_depthdata = new XnPoint3D[307200];
	pImageMap = new XnRGB24Pixel[307200];
	eResult = (*data).WaitNoneUpdateAll();
	if(eResult == XN_STATUS_OK )
	{	
		pDepthMap = mDepthGenerator.GetDepthMap();
		const XnRGB24Pixel* tmpImageMap = mImageGenerator.GetRGB24ImageMap();
		memcpy(pImageMap, tmpImageMap, sizeof(XnRGB24Pixel)*307200);
		
		mDepthGenerator.GetMetaData(mDepthMetaData);
		memcpy(imgDepth16u.data,mDepthMetaData.Data(),640*480*2);
		imgDepth16u.convertTo(depthshow,CV_8U,255/4096.0);
		depth_gray2rainbow(depthshow);
		
		mImageGenerator.GetMetaData(mImageMetaData);
		memcpy(imgRGB8u.data,mImageMetaData.Data(),640*480*3);
		cvtColor(imgRGB8u,imageshow,CV_RGB2BGR);
		
		colordepthshow = depth_gray2rainbow(depthshow);
		maskshow = mask_generate(colordepthshow, imageshow);
		meta2xnpoint(mDepthMetaData);
//		color_cloud = pointcloud_generate(&mDepthGenerator, &mDepthMetaData, pDepthMap, pImageMap);

		imshow("Depth", depthshow);
		imshow("ColorDepth",colordepthshow);
		imshow("Image", imageshow);
		imshow("Mask", maskshow);
//		waitKey(50);
	}
	else
	{
		cout << "Read Failed" << endl;
	}
}

void Kinect_Input::start()
{
	context_initial();
	mContext.SetGlobalMirror(true);
	
	depth_create(&mContext);
	image_create(&mContext);
	mapmode_setup(&mapMode);
	viewpoint_correct(&mImageGenerator);
	generatordata_start(&mContext);
//#####	int threadwhether=pthread_create(&thread1,NULL,Kinect_Input::pointcloud_thread,this);

}

void Kinect_Input::shut_down(Context* stop)
{
	(*stop).StopGeneratingAll();
	(*stop).Release();
}

Mat Kinect_Input::depth_gray2rainbow(Mat gray)
{
	Mat color(480,640,CV_8UC3);
	uchar temp;
	
	#define IMG_B(y,x) color.at<Vec3b>(y,x)[0]
	#define IMG_G(y,x) color.at<Vec3b>(y,x)[1]
	#define IMG_R(y,x) color.at<Vec3b>(y,x)[2]
	
	for(int i = 0; i < gray.rows; i++)
	{
		for(int j = 0; j < gray.cols; j++)
		{
			temp = gray.at<uchar>(i, j);
			
			if (temp <= 51)
			 {
			    IMG_B(i, j) = 255;
			    IMG_G(i, j) = temp*5;
			    IMG_R(i, j) = 0;
			 }
			 else if (temp <= 102)
			 {
			    temp-=51;
			    IMG_B(i, j) = 255-temp*5;
			    IMG_G(i, j) = 255;
			    IMG_R(i, j) = 0;
			 }
			 else if (temp <= 153)
			 {
			    temp-=102;
			    IMG_B(i, j) = 0;
			    IMG_G(i, j) = 255;
			    IMG_R(i, j) = temp*5;
			 }
			 else if (temp <= 204)
			 {
			    temp-=153;
			    IMG_B(i, j) = 0;
			    IMG_G(i, j) = 255-uchar(128.0*temp/51.0+0.5);
			    IMG_R(i, j) = 255;
			 }
			 else
			 {
			    temp-=204;
			    IMG_B(i, j) = 0;
			    IMG_G(i, j) = 127-uchar(127.0*temp/51.0+0.5);
			    IMG_R(i, j) = 255;
			 }
		}
	}

	return(color);
}

Mat Kinect_Input::mask_generate(Mat rainbow, Mat color)
{
	Mat mask(480,640,CV_8UC3);
	
	addWeighted(color, 1.0, rainbow, 0.4, 0, mask);
	
	return(mask);
}

void Kinect_Input::meta2xnpoint(DepthMetaData &m2x_DepthMetaData)
{
	int serialy, serial;
	int pointnumber = m2x_DepthMetaData.FullXRes()*m2x_DepthMetaData.FullYRes();
//	XnPoint3D* m2x_depthdata = new XnPoint3D[pointnumber];
	
	for(int j = 0; j < m2x_DepthMetaData.FullYRes(); ++j)
	{
		serialy = j * m2x_DepthMetaData.FullXRes();

		for(int i = 0; i < m2x_DepthMetaData.FullXRes(); ++i)
		{
			serial = serialy + i; 
			m_depthdata[serial].X = i;
			m_depthdata[serial].Y = j;
			m_depthdata[serial].Z = m2x_DepthMetaData(i, j);
		}
	}
//	return(m2x_depthdata);
}
#if 0
void* Kinect_Input::pointcloud_thread(void* pvData)
{
	Kinect_Input *ki = (Kinect_Input*) pvData;

	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(m_cloud);
	viewer.setBackgroundColor (255, 255, 255);
	viewer.addPointCloud<pcl::PointXYZRGB> (m_cloud, rgb, "sample cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer.addCoordinateSystem (2.0);

	while(1)
	{
//		(*ki).color_cloud = (*ki).pointcloud_generate(&((*ki).mDepthGenerator), &((*ki).mDepthMetaData), (*ki).pDepthMap, (*ki).pImageMap);
		m_cloud = (*ki).pointcloud_generate(&((*ki).mDepthGenerator), &((*ki).mDepthMetaData), (*ki).pDepthMap, (*ki).pImageMap);

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(m_cloud);	
		viewer.updatePointCloud(m_cloud, rgb, "sample cloud");
		viewer.spinOnce (100);
		if(viewer.wasStopped ())
		{
			cout << "FUCK" << endl;
			break;
		}

		usleep(100);
	}
}
#endif
#if 0
PointCloud<pcl::PointXYZRGB>::Ptr Kinect_Input::pointcloud_generate(DepthGenerator* depth, DepthMetaData* depthmeta, const XnDepthPixel* depthpixel, XnRGB24Pixel* imgpixel)
{
	int serialy=0;
	int serial=0; 
	int pointnumber = (*depthmeta).FullXRes()*(*depthmeta).FullYRes();
	XnPoint3D* depthpointcloud = new XnPoint3D[pointnumber];
	XnPoint3D* realpointcloud = new XnPoint3D[pointnumber];
//	vector<PointCloud_Color> temp_color_cloud;
	PointXYZRGB pclpoint;
	PointCloud<pcl::PointXYZRGB>::Ptr pclcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	
	for(int j = 0; j < (*depthmeta).FullYRes(); ++j)
	{
		serialy = j * (*depthmeta).FullXRes();

		for(int i = 0; i < (*depthmeta).FullXRes(); ++i)
		{
			serial = serialy + i; 
			depthpointcloud[serial].X = i;
			depthpointcloud[serial].Y = j;
			depthpointcloud[serial].Z = depthpixel[serial];
//			printout(depthpointcloud, serial);
		}
	}
	
	(*depth).ConvertProjectiveToRealWorld(pointnumber, depthpointcloud, realpointcloud);
//	for(int i=0; i<pointnumber; i++)
//	printout(realpointcloud, i);
	
	for(int i = 0; i < pointnumber; i++)
	{
		if(realpointcloud[i].Z == 0)
		{
//			temp_color_cloud.push_back(PointCloud_Color(realpointcloud[i], 0, 250, 154));
//			printout(temp_color_cloud, i);
			continue;
		}
		
		pclpoint.x = realpointcloud[i].X;
		pclpoint.y = realpointcloud[i].Y;
		pclpoint.z = realpointcloud[i].Z;
		pclpoint.r = imgpixel[i].nRed;
		pclpoint.g = imgpixel[i].nGreen;
		pclpoint.b = imgpixel[i].nBlue;
//		printout(pclpoint);
		pclcloud->points.push_back(pclpoint);
		
	}
	
	pclcloud->width = (int) pclcloud->points.size ();
	pclcloud->height = 1;

	delete depthpointcloud;
	delete realpointcloud;

	return(pclcloud);
}
#endif

void Kinect_Input::printout(vector<PointCloud_Color> printpoint, int num)
{
	cout <<"No. "  <<num<<"::	";			
	cout << "X:  " << printpoint[num].X << "	";  
	cout << "Y:  " << printpoint[num].Y << "	";   
	cout << "Z:  " << printpoint[num].Z << "	";   
	cout << "R:  " << printpoint[num].R << "	";   
	cout << "G:  " << printpoint[num].G << "	";   
	cout << "B:  " << printpoint[num].B <<endl;
}

void Kinect_Input::printout(XnPoint3D* printpoint, int num)
{
	cout <<"No. "  <<num<<"::	";			
	cout << "X:  " << printpoint[num].X << "	";  
	cout << "Y:  " << printpoint[num].Y << "	";   
	cout << "Z:  " << printpoint[num].Z <<endl;
}

void Kinect_Input::printout(PointXYZRGB printpoint)
{
//	cout <<"No. "  <<num<<"::	";			
	cout << "X:  " << printpoint.x << "	";  
	cout << "Y:  " << printpoint.y << "	";   
	cout << "Z:  " << printpoint.z << "	";   
	cout << "R:  " << (float)printpoint.r << "	";   
	cout << "G:  " << (float)printpoint.g << "	";   
	cout << "B:  " << (float)printpoint.b <<endl;
}

#include "Kinect_Input.h"
#include "Detect.h"
#include "PointCloudProcess.h"
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <iostream>

using namespace xn;
using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;

bool running = true, state1, state2;

Kinect_Input ki;

Detect det;

PointCloudProcess pcp;

extern PointCloud<pcl::PointXYZRGB>::Ptr g_cloud;
extern g2o::SparseOptimizer m_globalOptimizer;

void clear(int signo)
{
    std::cout << "exit....." << std::endl;
    running = false;
    while(state1);
    std::cout << "down!\n" << std::endl;;
    exit(0);
}

#if 1	
void vector_init()
{	
	for(int i = 0; i < 90; i++)
	{
		ki.data_read(&(ki.mContext));
	}
	
	vector<KeyPoint> initkeypoint = det.kp_extract(det.m_detector, ki.imageshow);
	Mat init_descriptor = det.descriptor_compute(det.m_detector, ki.imageshow, initkeypoint);
	
	Mat first_descriptor  = cv::Mat();
	init_descriptor.copyTo(first_descriptor);
	
	det.m_kpkeyframe.push_back(initkeypoint);
	det.m_deskeyframe.push_back(first_descriptor);
//	det.m_imgkeyframe.push_back(first_imgrecent);
	det.m_depthkeyframe.push_back(ki.m_depthdata);
	det.m_colorkeyframe.push_back(ki.pImageMap);
	
	det.init_g2o();
	//det.m_detflag = 1;
	det.inliers = Mat(10,10,CV_64F);
	pcp.map_generation(det.m_parament, det.inliers, ki.mDepthGenerator, det.m_depthkeyframe, det.m_colorkeyframe, 1);
}
#endif

#if 0
void test()
{
	Isometry3d T = Eigen::Isometry3d::Identity();
	T(0,0)=1;
	T(0,1)=0;
	T(0,2)=0;
	T(1,0)=0;
	T(1,1)=0.707;
	T(1,2)=-0.707;
	T(2,0)=0;
	T(2,1)=0.707;
	T(2,2)=0.707;
	T(0,3)=0;
	T(1,3)=0;
	T(2,3)=0.5;

	cout<<T.matrix()<<endl;
	
	for (int i = 2; i <=8; i++)
	{
		g2o::EdgeSE3* edge = new g2o::EdgeSE3();
		g2o::VertexSE3 *v = new g2o::VertexSE3();
		
		v->setId(i);
		v->setEstimate( Eigen::Isometry3d::Identity() );
	        m_globalOptimizer.addVertex(v);
	        
	        edge->vertices() [0] = m_globalOptimizer.vertex( i-1 );
	        edge->vertices() [1] = m_globalOptimizer.vertex( i );
	        
	        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
	        information(0,0) = information(1,1) = information(2,2) = 100;
	        information(3,3) = information(4,4) = information(5,5) = 100;
	        edge->setInformation( information );
	        edge->setMeasurement( T );
	        m_globalOptimizer.addEdge(edge);

		
		
	}
	
#if 0
	T = Eigen::Isometry3d::Identity();
	//T(2,3)=1;
	//cout<<T.matrix()<<endl;
	
	g2o::EdgeSE3* edge = new g2o::EdgeSE3();
		g2o::VertexSE3 *v = new g2o::VertexSE3();
		
		v->setId(8);
		v->setEstimate( Eigen::Isometry3d::Identity() );
	        m_globalOptimizer.addVertex(v);
	        
	        edge->vertices() [0] = m_globalOptimizer.vertex( 1 );
	        edge->vertices() [1] = m_globalOptimizer.vertex( 8 );
	        
	        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
	        information(0,0) = information(1,1) = information(2,2) = 100;
	        information(3,3) = information(4,4) = information(5,5) = 100;
	        edge->setInformation( information );
	        edge->setMeasurement( T );
	        m_globalOptimizer.addEdge(edge);
	

#endif
	
	m_globalOptimizer.save("pa.g2o");
	m_globalOptimizer.initializeOptimization();
	//m_globalOptimizer.optimize(50);
	m_globalOptimizer.save("pa2.g2o");
	cout<<"fuck"<<endl;
	VertexSE3* vertex = dynamic_cast<VertexSE3*>(m_globalOptimizer.vertex(2));
	Isometry3d pose = vertex -> estimate();
	//cout << "matrix:" << endl;
	cout << pose.matrix() << endl;
	
}
#endif

int main()
{
    signal(SIGINT, clear);
    signal(SIGTERM, clear);

    running = true;

	//test();
#if 1
	ki.start();
	vector_init();
	
	while(running)
	{
		cout << "************** Start **************" << endl;
		ki.data_read(&(ki.mContext));
		det.detect_process(ki.imageshow, ki.mDepthGenerator, ki.m_depthdata, ki.pImageMap);
		pcp.map_generation(det.m_parament, det.inliers, ki.mDepthGenerator, det.m_depthkeyframe, det.m_colorkeyframe, det.m_detflag);
		
//		if (pcp.m_flag4det == 0)
//			det.m_detflag = 0;
			
		waitKey(1000);
	}
	ki.shut_down(&(ki.mContext));
#endif
return 0;
}

#ifndef DETECT_H
#define DETECT_H

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector> 
#include <cmath>
#include <ctime>

#include <XnCppWrapper.h>
//#include <XnTypes.h>
#include <Kinect_Input.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/types/slam3d/types_slam3d.h> //顶点类型
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/marginal_covariance_cholesky.h>

using namespace xn;
using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;
using namespace g2o;

g2o::SparseOptimizer m_globalOptimizer;

class Detect
{
	public:
		
		Ptr<xfeatures2d::SURF> m_detector;
//		Ptr<DescriptorExtractor> descriptor;
		BFMatcher m_matcher;

		DepthMetaData m_DepthMeta_last;
		XnRGB24Pixel* m_colorpixel;
				
		vector<KeyPoint> m_lastkp;
		vector<KeyPoint> m_recentkp;
		vector<DMatch> m_matchpoint;

		std::vector<cv::DMatch> m_goodMatches;
		std::vector<cv::KeyPoint> m_goodObjectKeypoints;
		std::vector<cv::KeyPoint> m_goodSceneKeypoints;
		std::vector<cv::Point2f> m_goodScenePoints;
		double m_goodMatchMinValue;
		double m_goodMatchDistanceTimes;

		cv::Mat rvec, tvec, inliers;
		cv::Mat cameraMatrix;

		XnPoint3D* m_depthlast;
		Mat m_imglast;
		Mat m_imgrecent;
		Mat m_deplast;
		Mat m_deprecent;
		Mat m_descriptorlast;
		Mat m_descriptorrecent;
		
 		Isometry3d m_parament;
 		vector<vector<KeyPoint>> m_kpkeyframe;
 		vector<Mat> m_deskeyframe;
 	//	vector<Mat> m_imgkeyframe;
 		vector<XnPoint3D*> m_depthkeyframe;
 		vector<XnRGB24Pixel*> m_colorkeyframe;
		int m_detflag;
		
		double m_kfthreshold;
		enum situation{	NOT_MATCED = 0,
				TOO_FAR_AWAY,
				TOO_CLOSE,
				KEYFRAME
		   	      };
		
	public:
		Detect();
		~Detect();
		void init_g2o();
		vector<cv::KeyPoint> kp_extract(Ptr<xfeatures2d::SURF> det, Mat img);
		Mat descriptor_compute(Ptr<xfeatures2d::SURF> det, Mat img, vector<cv::KeyPoint> kp);
		vector<DMatch> img_match(BFMatcher match, Mat last, Mat recent);
		void surf_show(Mat last, Mat recent, vector<KeyPoint> kpl, vector<KeyPoint> kpr, vector<DMatch> points);
		bool getGoodMatchesA(vector<DMatch>& in_match, vector<DMatch>& in_goodMatch);
		bool pointcloud_generation(uint32_t in_size, XnPoint3D* in_depthCloud, XnPoint3D* in_realCloud);
		Isometry3d matrix_generation(Mat mg_rvec, Mat mg_tvec);
		double normofTransform(Mat nmt_rvec, Mat nmt_tvec );
		void detect_process(Mat dp_image, DepthGenerator dp_depth, XnPoint3D* dp_depthrecent, XnRGB24Pixel* dp_colorpix);
		void processing(vector<KeyPoint> pr_kplast, Mat m_descriptorlast, Mat pr_imgrecent, XnPoint3D* pr_depthlast, XnPoint3D* pr_depthrecent, DepthGenerator pr_depth, int pr_selectedkeyframe, bool pr_ifloop = false);
		void ransac_detect(vector<DMatch> rd_matchpoint, vector<KeyPoint> rd_lastkp, vector<KeyPoint> rd_recentkp, Mat rd_descriptorrecent, DepthGenerator rd_depth, XnPoint3D* rd_depthlast, XnPoint3D* rd_depthrecent, Mat rd_imgrecent, int rd_selectedkeyframe, bool ifloop = false);
		
	public:
		typedef BlockSolver_6_3 SlamBlockSolver; 
 		typedef LinearSolverCSparse< BlockSolver_6_3::PoseMatrixType > SlamLinearSolver;
 		
 		SlamLinearSolver* linearSolver;
 		SlamBlockSolver* blockSolver;
 		OptimizationAlgorithmLevenberg* solver;
/////// 		SparseOptimizer m_globalOptimizer; 
 		VertexSE3* v;
 		Eigen::Matrix<double, 6, 6> information;
 		
 	public:
 		void adjacent_loopdetect(vector<vector<KeyPoint>> alp_kpkeyframe, vector<Mat> alp_deskeyframe, Mat alp_imgrecent, vector<XnPoint3D*> alp_depthlast, XnPoint3D* alp_depthrecent, DepthGenerator alp_depth, bool alp_ifloop = true);
 		void random_loopdetect(vector<vector<KeyPoint>> rlp_kpkeyframe, vector<Mat> rlp_deskeyframe, Mat rlp_imgrecent, vector<XnPoint3D*> rlp_depthlast, XnPoint3D* rlp_depthrecent, DepthGenerator rlp_depth, bool rlp_ifloop = true);
};
#endif

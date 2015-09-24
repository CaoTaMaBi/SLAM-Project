#ifndef DETECT_H
#define DETECT_H

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector> 

#include <Kinect_Input.h>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

class Detect
{
	public:
		
		Ptr<xfeatures2d::SURF> m_detector;
//		Ptr<DescriptorExtractor> descriptor;
		BFMatcher m_matcher;

		DepthMetaData m_DepthMeta_last;
				
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

		Mat m_imglast;
		Mat m_imgrecent;
		Mat m_deplast;
		Mat m_deprecent;
		Mat m_descriptorlast;
		Mat m_descriptorrecent;
		int fuck;
		
	public:
		Detect();
		vector<cv::KeyPoint> kp_extract(Ptr<xfeatures2d::SURF> det, Mat img);
		Mat descriptor_compute(Ptr<xfeatures2d::SURF> det, Mat img, vector<cv::KeyPoint> kp);
		vector<DMatch> img_match(BFMatcher match, Mat last, Mat recent);
		void ransac_detect(vector<DMatch> tmp_matchpoint, vector<KeyPoint> tmp_lastkp, vector<KeyPoint> tmp_recentkp, DepthGenerator tmp_depth, DepthMetaData &tmp_depthmeta);//, const XnDepthPixel* tmp_depthpixel);
		void surf_show(Mat last, Mat recent, vector<KeyPoint> kpl, vector<KeyPoint> kpr, vector<DMatch> points);
		void surf_process(Mat tmp_image, DepthGenerator tmp_depth, DepthMetaData &tmp_depthmeta, const XnDepthPixel* tmp_depthpixel);
		bool getGoodMatchesA();
};
#endif

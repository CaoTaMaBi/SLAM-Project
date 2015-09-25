#include <Detect.h>

using namespace std;
using namespace cv;

Detect::Detect()
{
	m_detector = xfeatures2d::SURF::create();
//	descriptor = DescriptorExtractor::create("SURF");

	m_imglast.create(480,640,CV_8UC3);
	m_imgrecent.create(480,640,CV_8UC3);
	m_deplast.create(480,640,CV_8UC1);
	m_deprecent.create(480,640,CV_8UC3);
//	m_descriptorlast.create(480,640,CV_8UC3);
//	m_descriptorrecent.create(480,640,CV_8UC3);

	m_goodMatchMinValue = 0.002;
	m_goodMatchDistanceTimes = 3;

	double cx = 320;
	double cy = 240;
	double fx = 525;
	double fy = 525;
		
	cameraMatrix = cv::Mat(3, 3, CV_64F);
	cameraMatrix.setTo(0);
	cameraMatrix.at<double>(0,0) = fx;
	cameraMatrix.at<double>(0,2) = cx;
	cameraMatrix.at<double>(1,1) = fy;
	cameraMatrix.at<double>(1,2) = cy;
	cameraMatrix.at<double>(2,2) = 1;

	namedWindow("KeyPoint_Show");
	namedWindow("Match_Show");

}

vector<cv::KeyPoint> Detect::kp_extract(Ptr<xfeatures2d::SURF> det, Mat img)
{
	vector<cv::KeyPoint> tmp;
	
	det->detect(img, tmp);
	
	return(tmp);	
}

Mat Detect::descriptor_compute(Ptr<xfeatures2d::SURF> det, Mat img, vector<cv::KeyPoint> kp)
{
	Mat tmp;
	
	det->compute(img, kp, tmp);
	
	return(tmp);
}		

vector<DMatch> Detect::img_match(BFMatcher matcher, Mat last, Mat recent)
{
	vector<DMatch> tmp;
	
	matcher.match(last, recent, tmp);
	
	return(tmp);
}

bool Detect::getGoodMatchesA()
{
	m_goodMatches.clear();
	m_goodObjectKeypoints.clear();
	m_goodSceneKeypoints.clear();

	//Calcmm_lastkp_lastkpulate closest match
	double minMatchDis = 9999;
	size_t minMatchIndex = 0;
	if(m_matchpoint.size() == 0)
	{
		std::cout << "m_matchpoint is empty" << std::endl;
		return false;
	}

	for ( size_t i=0; i<m_matchpoint.size(); i++ )
	{
		if ( m_matchpoint[i].distance < minMatchDis )
		{
			minMatchDis = m_matchpoint[i].distance;
			minMatchIndex = i;
		}
	}

	if(m_goodMatchDistanceTimes * minMatchDis > m_goodMatchMinValue)
	{
		std::cout << "use m_goodMatchDistanceTimes" << std::endl;
	}
	else
	{
		std::cout << "use m_goodMatchMinValue" << std::endl;
	}
	double maxDistance = std::max(m_goodMatchDistanceTimes * minMatchDis, m_goodMatchMinValue);
	for (size_t i=0; i<m_matchpoint.size(); i++ )
	{
		if(m_matchpoint[i].distance <= maxDistance)
		{
			m_goodMatches.push_back(m_matchpoint[i]);
			m_goodObjectKeypoints.push_back(m_lastkp[m_matchpoint[i].queryIdx]);
			m_goodSceneKeypoints.push_back(m_recentkp[m_matchpoint[i].trainIdx]);
		}
	}

	if(m_goodMatches.size() == 0)
	{
		std::cout << "m_goodMatches is empty" << std::endl;
		return false;
	}

	return true;
}

void Detect::surf_show(Mat last, Mat recent, vector<KeyPoint> kpl, vector<KeyPoint> kpr, vector<DMatch> points)
{
	Mat kp_show, match_show;
	
//	drawKeypoints(last, kpl, kp_show, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(recent, kpr, kp_show, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawMatches(last, kpl, recent, kpr, points, match_show);
	
	imshow("KeyPoint_Show", kp_show);
	imshow("Match_Show", match_show);
	waitKey(10);
}

void Detect::ransac_detect(vector<DMatch> tmp_matchpoint, vector<KeyPoint> tmp_lastkp, vector<KeyPoint> tmp_recentkp, DepthGenerator tmp_depth, DepthMetaData &tmp_depthmeta)
{
	vector<Point3f> pts_last;
	vector<Point2f> pts_recent;
	Point2f temp2;
	Point3f temp;
	XnPoint3D* depthpointcloud = new XnPoint3D[tmp_matchpoint.size()];
	XnPoint3D* realpointcloud = new XnPoint3D[tmp_matchpoint.size()];
	
	for(int i = 0; i < tmp_matchpoint.size(); i++)
	{
		depthpointcloud[i].X = tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.x;
		depthpointcloud[i].Y = tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.y;
		depthpointcloud[i].Z = tmp_depthmeta((int) tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.x, (int) tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.y);
		
	//	cout << "X: " << depthpointcloud[i].X<<"  ";
	//	cout << "Y: " << depthpointcloud[i].Y<<"  ";
	//	cout << "Z: " << depthpointcloud[i].Z<<endl;
	}
	
	tmp_depth.ConvertProjectiveToRealWorld(tmp_matchpoint.size(), depthpointcloud, realpointcloud);
	
	cout<<"aaaaaaa"<<tmp_matchpoint.size()<<endl;
	for(int i = 0; i < tmp_matchpoint.size(); i++)
	{
		if (realpointcloud[i].Z == 0)
			continue;
		
		temp2.x = tmp_recentkp[tmp_matchpoint[i].trainIdx].pt.x;
		temp2.y = tmp_recentkp[tmp_matchpoint[i].trainIdx].pt.y;
		pts_recent.push_back(temp2);
//		pts_recent.push_back(Point2f(tmp_recentkp[tmp_matchpoint[i].trainIdx].pt));

		temp.z = realpointcloud[i].Z;
		temp.x = realpointcloud[i].X;
		temp.y = realpointcloud[i].Y;

		pts_last.push_back(temp);
	}

	// 求解pnp
	bool result;

	result = solvePnPRansac( pts_last, pts_recent, cameraMatrix, Mat(), rvec, tvec);
	cout << result << endl;

	std::cout << "rvec:" << std::endl;
	std::cout << rvec << std::endl;
	std::cout << "tvec:" << std::endl;
	std::cout << tvec << std::endl;

}

void Detect::surf_process(Mat tmp_image, DepthGenerator tmp_depth, DepthMetaData &tmp_depthmeta, const XnDepthPixel* tmp_depthpixel)
{	
	tmp_image.copyTo(m_imgrecent);

	m_lastkp = kp_extract(m_detector, m_imglast);
	m_recentkp = kp_extract(m_detector, m_imgrecent);
	
	m_descriptorlast = descriptor_compute(m_detector, m_imglast, m_lastkp);
	m_descriptorrecent = descriptor_compute(m_detector, m_imgrecent, m_recentkp);
	
	m_matchpoint = img_match(m_matcher, m_descriptorlast, m_descriptorrecent);
//	cout << "matchpoint size:    " << m_matchpoint.size() << endl;
	
	if(m_matchpoint.size()>100)
	{
		ransac_detect(m_matchpoint, m_lastkp, m_recentkp, tmp_depth, m_DepthMeta_last);
	}
	
	surf_show(m_imglast, m_imgrecent, m_lastkp, m_recentkp, m_matchpoint);
	m_imgrecent.copyTo(m_imglast);
	m_DepthMeta_last.CopyFrom(tmp_depthmeta);	
}

#include <Detect.h>

using namespace xn;
using namespace cv;
using namespace std;
using namespace pcl;
using namespace Eigen;
using namespace g2o;

///g2o::SparseOptimizer m_globalOptimizer;

Detect::Detect()
{
    //cout << "catamabi" << endl;
    m_detflag = 0;
    
    m_detector = xfeatures2d::SURF::create(5000);
//  descriptor = DescriptorExtractor::create("SURF");

    m_imglast.create(480,640,CV_8UC3);
    m_imgrecent.create(480,640,CV_8UC3);
    m_deplast.create(480,640,CV_8UC1);
    m_deprecent.create(480,640,CV_8UC3);

    m_goodMatchMinValue = 0.002;
    m_goodMatchDistanceTimes = 4;

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
    
    m_parament = Isometry3d::Identity();

    
    
    m_kfthreshold = 0.2;

    namedWindow("KeyPoint_Show");
    namedWindow("Match_Show");
    
    init_g2o();
    information = Eigen::Matrix<double, 6, 6>::Identity();
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;

}

Detect::~Detect()
{
    m_globalOptimizer.save("pa.g2o");
}

void Detect::init_g2o()
{
    linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    blockSolver = new SlamBlockSolver( linearSolver );
    solver = new OptimizationAlgorithmLevenberg( blockSolver );
    m_globalOptimizer.setAlgorithm( solver );
    m_globalOptimizer.setVerbose( false );
    
    v = new VertexSE3();
    v->setId( 1 );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    v->setFixed( true );
    m_globalOptimizer.addVertex( v );
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
    vector<DMatch> tmp, tmp2;
    matcher.match(last, recent, tmp);
    getGoodMatchesA(tmp, tmp2);
    
    return(tmp2);
}

bool Detect::getGoodMatchesA(vector<DMatch>& in_match, vector<DMatch>& in_goodMatch)
{
    //m_goodMatches.clear();
    m_goodObjectKeypoints.clear();
    m_goodSceneKeypoints.clear();

    //Calcmm_lastkp_lastkpulate closest match
    double minMatchDis = 9999;
    size_t minMatchIndex = 0;
    if(in_match.size() == 0)
    {
        std::cout << "m_matchpoint is empty" << std::endl;
        return false;
    }

    for ( size_t i=0; i<in_match.size(); i++ )
    {
        //cout << "Distance" << in_match[i].distance << endl;
        if ( in_match[i].distance < minMatchDis )
        {
            minMatchDis = in_match[i].distance;
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
    for (size_t i=0; i<in_match.size(); i++ )
    {
        if(in_match[i].distance <= maxDistance)
        {
            in_goodMatch.push_back(in_match[i]);
            m_goodObjectKeypoints.push_back(m_lastkp[in_match[i].queryIdx]);
            m_goodSceneKeypoints.push_back(m_recentkp[in_match[i].trainIdx]);
        }
    }

    if(in_goodMatch.size() == 0)
    {
        std::cout << "m_goodMatches is empty" << std::endl;
        return false;
    }
    
//  cout << "#####MinValue#####" << minMatchDis << endl;
//  cout << "#####Match Size##### " << in_match.size() << endl;
    cout << "GoodMatch Size: " << in_goodMatch.size() << endl;

    return true;
}

void Detect::surf_show(Mat last, Mat recent, vector<KeyPoint> kpl, vector<KeyPoint> kpr, vector<DMatch> points)
{
    Mat kp_show, match_show;
    
//  drawKeypoints(last, kpl, kp_show, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    //drawKeypoints(recent, kpr, kp_show, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawMatches(last, kpl, recent, kpr, points, match_show);
    
    //imshow("KeyPoint_Show", kp_show);
    imshow("Match_Show", match_show);
    waitKey(10);
}

bool Detect::pointcloud_generation(uint32_t in_size, XnPoint3D* in_depthCloud, XnPoint3D* in_realCloud)
{
    double cx = 320;
    double cy = 240;
    double fx = 525;
    double fy = 525;
    double scale = 1000;

    for (uint32_t i = 0; i < in_size; i++)
    {
        in_realCloud[i].Z = double(in_depthCloud[i].Z / scale);
        in_realCloud[i].X = (in_depthCloud[i].X - cx) * in_realCloud[i].Z / fx;
        in_realCloud[i].Y = (in_depthCloud[i].Y - cy) * in_realCloud[i].Z / fy;
        //std::cout << "TTTTTTTTTTTTTT" << i << ": " << in_depthCloud[i].X << ", " << in_depthCloud[i].Y << ", " << in_depthCloud[i].Z << std::endl;
    }
    return true;
}

void Detect::ransac_detect(vector<DMatch> rd_matchpoint, vector<KeyPoint> rd_lastkp, vector<KeyPoint> rd_recentkp, Mat rd_descriptorrecent, DepthGenerator rd_depth, XnPoint3D* rd_depthlast, XnPoint3D* rd_depthrecent, Mat rd_imgrecent, int rd_selectedkeyframe, bool ifloop)
{
    static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Cauchy" );
    //std::cout << "rd_depthrecent: " << rd_depthrecent << std::endl;
    //std::cout << "rd_depthlast: " << rd_depthlast << std::endl;
    double norm;
    vector<Point3f> pts_last;
    vector<Point2f> pts_recent;
    Point2f temp2;
    Point3f temp;
    XnPoint3D* depthpointcloud = new XnPoint3D[rd_matchpoint.size()];
    XnPoint3D* realpointcloud = new XnPoint3D[rd_matchpoint.size()];

    VertexSE3* rd_v = new g2o::VertexSE3();
    EdgeSE3* rd_edge = new g2o::EdgeSE3();
//  Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
#if 0   
    for(int i = 0; i < tmp_matchpoint.size(); i++)
    {
        depthpointcloud[i].X = tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.x;
        depthpointcloud[i].Y = tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.y;
        depthpointcloud[i].Z = rd_lastdepthmeta((int) tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.x, (int) tmp_lastkp[tmp_matchpoint[i].queryIdx].pt.y);
        
    //  cout << "X: " << depthpointcloud[i].X<<"  ";
    //  cout << "Y: " << depthpointcloud[i].Y<<"  ";
    //  cout << "Z: " << depthpointcloud[i].Z<<endl;
    }
#endif  
    for(int i = 0; i < rd_matchpoint.size(); i++)
    {

        depthpointcloud[i].X = rd_lastkp[rd_matchpoint[i].queryIdx].pt.x;
        depthpointcloud[i].Y = rd_lastkp[rd_matchpoint[i].queryIdx].pt.y;
        depthpointcloud[i].Z = rd_depthlast[((int)(rd_lastkp[rd_matchpoint[i].queryIdx].pt.y)*640)+(int)(rd_lastkp[rd_matchpoint[i].queryIdx].pt.x)].Z;
        //depthpointcloud[i].Z = (-1) * depthpointcloud[i].Z;
    }

    //rd_depth.ConvertProjectiveToRealWorld(rd_matchpoint.size(), depthpointcloud, realpointcloud);
    pointcloud_generation(rd_matchpoint.size(), depthpointcloud, realpointcloud);
    for(int i = 0; i < rd_matchpoint.size(); i++)
    {
        if (realpointcloud[i].Z == 0)
            continue;
        temp2.x = rd_recentkp[rd_matchpoint[i].trainIdx].pt.x;
        temp2.y = rd_recentkp[rd_matchpoint[i].trainIdx].pt.y;
        pts_recent.push_back(temp2);
//      pts_recent.push_back(Point2f(tmp_recentkp[tmp_matchpoint[i].trainIdx].pt));

        temp.z = realpointcloud[i].Z;
        temp.x = realpointcloud[i].X;
        temp.y = realpointcloud[i].Y;
        
        pts_last.push_back(temp);
    }

    // 求解pnp
    bool result;
    result = solvePnPRansac(pts_last, pts_recent, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers);//solvePnPRansac( pts_last, pts_recent, cameraMatrix, Mat(), rvec, tvec); //需要加inliers参数，为下一步判断提供数据
#if 0
    std::cout << "rvec:" << std::endl;
    std::cout << rvec << std::endl;
    std::cout << "tvec:" << std::endl;
    std::cout << tvec << std::endl;
    std::cout << "inliers: " << inliers.rows << std::endl;
#endif
#if 1           
    if (inliers.rows < 5)
    {
        cout << "Not Match" << endl;
        return;
    }
    norm = normofTransform(rvec, tvec);
    
    if(ifloop == false)
    {
        if (norm >= 0.4)
        {
            cout << "Too Far Away" << endl;
            return;
        }
    }
    else
    {
        if(norm >= 5)
        {
            cout << "Loop Too Far Away" << endl;
            return;
        }
    }
    
    if (norm <= m_kfthreshold)
    {
        cout << "Too Close" << endl;
        return;
    }
#endif  
    if(ifloop == false)
    {
//      m_imgkeyframe.push_back(rd_imgrecent);
        
        m_kpkeyframe.push_back(rd_recentkp);
        m_deskeyframe.push_back(rd_descriptorrecent);
        m_depthkeyframe.push_back(rd_depthrecent);
        m_colorkeyframe.push_back(m_colorpixel);
    }

    m_parament = matrix_generation(rvec, tvec);
    
    if(ifloop == false)
    {
        rd_v->setId(m_kpkeyframe.size());
        rd_v->setEstimate(Eigen::Isometry3d::Identity());
        m_globalOptimizer.addVertex(rd_v);
    }
    
    rd_edge->vertices()[0] = m_globalOptimizer.vertex(rd_selectedkeyframe);//(m_kpkeyframe.size() - 1);
    rd_edge->vertices()[1] = m_globalOptimizer.vertex(m_kpkeyframe.size());
    rd_edge->setRobustKernel(robustKernel);
    
//  information(0,0) = information(1,1) = information(2,2) = 100;
//  information(3,3) = information(4,4) = information(5,5) = 100;
    rd_edge->setInformation(information);
    rd_edge->setMeasurement(m_parament);
    m_globalOptimizer.addEdge(rd_edge);
    
//  m_imgrecent.copyTo(m_imglast);
//  m_DepthMeta_last.CopyFrom(rd_recentdepthmeta);

    
    //m_globalOptimizer.initializeOptimization();
    //m_globalOptimizer.optimize(50);

    if(ifloop == false)
    {
        //std::cout << "current vertex size: " <<  m_globalOptimizer.vertex().size();
        adjacent_loopdetect(m_kpkeyframe, m_deskeyframe, m_imgrecent, m_depthkeyframe, rd_depthrecent, rd_depth, true);
        random_loopdetect(m_kpkeyframe, m_deskeyframe, m_imgrecent, m_depthkeyframe, rd_depthrecent, rd_depth, true);
        m_detflag =1;
    }
    
    delete depthpointcloud;
    delete realpointcloud;

}

Isometry3d Detect::matrix_generation(Mat mg_rvec, Mat mg_tvec)
{
    Mat R;
    Matrix3d r;
    Isometry3d T = Eigen::Isometry3d::Identity();
    Isometry3d rT = Eigen::Isometry3d::Identity();

    std::cout << "rvec:" << std::endl;
    std::cout << mg_rvec << std::endl;
    std::cout << "tvec:" << std::endl;
    std::cout << mg_tvec << std::endl;

    mg_rvec = mg_rvec.t();
    //cout << mg_rvec.at<double>(0,1) << endl;


    Rodrigues(mg_rvec, R);
    //cout << R << endl;
    cv2eigen(R.t(), r);
    //cout << r.matrix() << endl;
    AngleAxisd angle(r);

    Translation<double,3> trans(mg_tvec.at<double>(0,0), mg_tvec.at<double>(1,0), mg_tvec.at<double>(2,0));
//  std::cout << "tvec:" << std::endl;
//  std::cout << mg_tvec << std::endl;  
    T = angle;
    //cout << angle.matrix() << endl;
    T(0,3) = mg_tvec.at<double>(0,0);
    T(1,3) = mg_tvec.at<double>(1,0);
    T(2,3) = mg_tvec.at<double>(2,0);

    //cout << mg_tvec.at<double>(0,0) << endl;
    //cout << mg_tvec.at<double>(0,1) << endl;
    //cout << mg_tvec.at<double>(0,2) << endl;
    
    rT = T.inverse();

    //cout << T.matrix() << endl;
    cout << rT.matrix() << endl;
    return (T);//这里到底用T还是rt
}

double Detect::normofTransform(Mat nmt_rvec, Mat nmt_tvec )
{
    return fabs(min(cv::norm(nmt_rvec), 2*M_PI-cv::norm(nmt_rvec)))+ fabs(cv::norm(nmt_tvec));
}

void Detect::adjacent_loopdetect(vector<vector<KeyPoint>> alp_kpkeyframe, vector<Mat> alp_deskeyframe, Mat alp_imgrecent, vector<XnPoint3D*> alp_depthlast, XnPoint3D* alp_depthrecent, DepthGenerator alp_depth, bool alp_ifloop)
{
    if(alp_kpkeyframe.size() <= 5)
    {
        for(size_t i = 0; i < alp_kpkeyframe.size()-2; i++)
        {
            processing(alp_kpkeyframe[i], alp_deskeyframe[i], alp_imgrecent, alp_depthlast[i], alp_depthrecent, alp_depth, i+1, alp_ifloop);
        }
    }
    else
    {
        for(size_t i = alp_kpkeyframe.size() - 5; i < alp_kpkeyframe.size()-2; i++ )
        {
            processing(alp_kpkeyframe[i], alp_deskeyframe[i], alp_imgrecent, alp_depthlast[i], alp_depthrecent, alp_depth, i+1, alp_ifloop);
        }
    }
}

void Detect::random_loopdetect(vector<vector<KeyPoint>> rlp_kpkeyframe, vector<Mat> rlp_deskeyframe, Mat rlp_imgrecent, vector<XnPoint3D*> rlp_depthlast, XnPoint3D* rlp_depthrecent, DepthGenerator rlp_depth, bool rlp_ifloop)
{
    srand((unsigned int) time(NULL));
    
    if(rlp_kpkeyframe.size() <= 5)
    {
        for(size_t i = 0; i < rlp_kpkeyframe.size()-2; i++)
        {
            processing(rlp_kpkeyframe[i], rlp_deskeyframe[i], rlp_imgrecent, rlp_depthlast[i], rlp_depthrecent, rlp_depth, i+1, rlp_ifloop);
        }
    }
    else
    {
        for(int i = 0; i < 5; i++)
        {
            int index = rand() % rlp_kpkeyframe.size();
            processing(rlp_kpkeyframe[index], rlp_deskeyframe[index], rlp_imgrecent, rlp_depthlast[index], rlp_depthrecent, rlp_depth, index+1, rlp_ifloop);
        }
    }
}

void Detect::processing(vector<KeyPoint> pr_kplast, Mat pr_descriptorlast, Mat pr_imgrecent, XnPoint3D* pr_depthlast, XnPoint3D* pr_depthrecent, DepthGenerator pr_depth, int pr_selectedkeyframe, bool pr_ifloop)
{
//  m_lastkp = kp_extract(m_detector, pr_imglast);
    m_recentkp = kp_extract(m_detector, pr_imgrecent);
    
//  m_descriptorlast = descriptor_compute(m_detector, pr_imglast, m_lastkp);
    m_descriptorrecent = cv::Mat(); 
    m_descriptorrecent = descriptor_compute(m_detector, pr_imgrecent, m_recentkp);
    
    m_matchpoint = img_match(m_matcher, pr_descriptorlast, m_descriptorrecent);
    
    //surf_show(m_imglast, m_imgrecent, m_lastkp, m_recentkp, m_matchpoint);
    
    if(m_matchpoint.size()>10)
    {
        ransac_detect(m_matchpoint, pr_kplast, m_recentkp, m_descriptorrecent, pr_depth, pr_depthlast, pr_depthrecent, pr_imgrecent, pr_selectedkeyframe, pr_ifloop);
    }
}

void Detect::detect_process(Mat dp_image, DepthGenerator dp_depth, XnPoint3D* dp_depthrecent, XnRGB24Pixel* dp_colorpix)
{   
    m_detflag = 0;
//  cout << &m_imgrecent.data << endl;
//  cout << &m_imglast.data << endl;

    m_imgrecent  = cv::Mat();
    dp_image.copyTo(m_imgrecent);
//  
//  m_imglast = m_imgkeyframe[m_imgkeyframe.size()-1];
    
//  m_descriptorlast = cv::Mat();

    //XnPoint3D* tempDepth3D = new XnPoint3D[307200];
    //memcpy(tempDepth3D, dp_depthrecent, sizeof(XnPoint3D)*307200);
    
    m_lastkp = m_kpkeyframe[m_kpkeyframe.size() - 1];
    m_deskeyframe[m_kpkeyframe.size() - 1].copyTo(m_descriptorlast);
    m_depthlast = m_depthkeyframe[m_kpkeyframe.size() - 1];
    
    //imshow("last", m_imglast);
    //imshow("recent",m_imgrecent);
    //waitKey(0);
    
//  for(int a = 1; a < m_imgkeyframe.size(); a++)
//  {
//      cout << a << endl;
//      imshow("keylast", m_imgkeyframe[a-1]);
//      imshow("key", m_imgkeyframe[a]);
//      waitKey(0);
//  }
    
    m_colorpixel = dp_colorpix;

    processing(m_lastkp, m_descriptorlast, m_imgrecent, m_depthlast, dp_depthrecent, dp_depth, m_kpkeyframe.size(), false);

    //adjacent_loopdetect(m_kpkeyframe, m_deskeyframe, m_imgrecent, m_depthkeyframe, dp_depthrecent, dp_depth, true);
    //random_loopdetect(m_kpkeyframe, m_deskeyframe, m_imgrecent, m_depthkeyframe, dp_depthrecent, dp_depth, true);
    //m_detflag =1;
    cout << "matchpoint size:    " << m_matchpoint.size() << endl;
        
}

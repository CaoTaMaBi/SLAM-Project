#include "Kinect_Input.h"
#include "Detect.h"
#include "PointCloudProcess.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace xn;
using namespace cv;

int main()
{
	Kinect_Input ki;
	Detect det;
	PointCloudProcess pcp;
	
	ki.start();
	while(1)
	{
		ki.data_read(&(ki.mContext));
		det.surf_process(ki.imageshow, ki.mDepthGenerator, ki.mDepthMetaData, ki.pDepthMap);
		pcp.map_generation(det.rvec, det.tvec, &(ki.mDepthGenerator), &(ki.mDepthMetaData), ki.pDepthMap, ki.pImageMap);
		waitKey(10);
	}
	ki.shut_down(&(ki.mContext));
	return 0;
}

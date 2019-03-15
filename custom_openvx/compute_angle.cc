#include <map>
#include "functions.hpp"
#include "common.hpp"
#include <opencv2/core/core.hpp>
#include <cmath>

void ComputeAngle::print_size_input() {
	std::cout << image->size_element() << std::endl;
	std::cout << in_kp->size_element() << std::endl;
}

void ComputeAngle::wait_output() {
	out_kp->waitTransfer();
}
void ComputeAngle::transfer_input() {
	image->wait(where);
	in_kp->wait(where);
}
void ComputeAngle::transfer_output() {
	out_kp->setWhere(where);
	out_kp->finishProcessing();
}

MeasureReturn ComputeAngle::execute_gpu() {
	//todo launch exception?
	this->where = CPU;
	execute_cpu();
	return MeasureReturn(-1);
}
void ComputeAngle::shouldTransferOutput( bool b ) {
	out_kp->setToBeTransferred( b );
}

static float IC_Angle(const cv::Mat& image, cv::Point2f pt,  const std::vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE_C; u <= HALF_PATCH_SIZE_C; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE_C; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);
}

MeasureReturn ComputeAngle::execute_cpu() {
	MeasureReturn m(currentFrameNumber);
	
	int init_inkp = in_kp->size();
	
	out_kp->setSize(0);
	for (int i = 0; i < in_kp->size(); i++)
    {
		cv::KeyPoint kp = in_kp->vec[i];
        kp.angle = IC_Angle(image->img, kp.pt, umax);
        out_kp->push_back(kp);
    }
    //std::cout << "IN_kp ANGLE " << level << ": " << in_kp->size() << std::endl;
    //std::cout << "OUT_kp ANGLE " << level << ": " << out_kp->size() << std::endl;
    
    //fprintf(stderr, "ANGLE: %d - %d\n", init_inkp, out_kp->size());
    
	return m;
}

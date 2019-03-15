#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/cudafeatures2d.hpp>

#include <iostream>

void fast_example(cv::Mat frame) {
cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(7, true, cv::FastFeatureDetector::TYPE_9_16);
cv::Ptr<cv::cuda::FastFeatureDetector> gpuFastDetector = cv::cuda::FastFeatureDetector::create(7, true, cv::FastFeatureDetector::TYPE_9_16, 70000);

std::vector<cv::KeyPoint> keypoints;
std::vector<cv::KeyPoint> gpuKeypoints;

cv::cuda::GpuMat gFrame;

gFrame.upload(frame);
std::cout << "**" << std::endl;
gpuFastDetector->detect(gFrame, gpuKeypoints);
std::cout << "FAST GPU " << gpuKeypoints.size() << std::endl;
std::cout << "@@" << std::endl;
fastDetector->detect(frame, keypoints);
std::cout << "FAST " << keypoints.size() << std::endl;
std::cout << "**" << std::endl;

}

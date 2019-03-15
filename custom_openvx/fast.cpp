#include "functions.hpp"
#include "common.hpp"
#include <opencv2/highgui/highgui.hpp>

#include "Viewer.h"

void FAST_kernel::print_size_input() {
	std::cout << src->size_element() << std::endl;
}

void FAST_kernel::wait_output() {
	keypoints->waitTransfer();
}
void FAST_kernel::transfer_input() {
	src->wait(where);
}
void FAST_kernel::transfer_output() {
	keypoints->setWhere(where);
	keypoints->finishProcessing();
	//this->wait_output();
	//int n = -1;
	//CUDA_SAFE_CALL( cudaMemcpy((&n), d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	//std::cout << "FAST " << level << " - sz: " << keypoints->size() << std::endl;
}

void FAST_kernel::shouldTransferOutput( bool b ) {
	keypoints->setToBeTransferred( b );
}

int calcKeypoints_gpu(const uchar* imgData, int cols, int rows, int step, short2* kpLoc, int maxKeypoints, int* score, int threshold, unsigned int* d_counter, cudaStream_t stream, cudaEvent_t computationFinished);
void nonmaxSuppression_gpu(int step, const short2* kpLoc, unsigned int* count, int* score, short2* loc, float* response, unsigned int* d_counter, cudaStream_t stream, cudaEvent_t computationFinished);
MeasureReturn FAST_kernel::execute_gpu() {
	MeasureReturn m(currentFrameNumber);
	
	keypoints->clear();
	keypoints_tmp.clear();
	
	cv::Mat img = src->img;
	
	uchar* d_img = src->d_img;
	short2* d_kp = keypoints_tmp.d_kp;
	short2* d_kp_final = keypoints->d_kp;
	
	float* d_response = keypoints->d_response;
	int* d_score = score->d_img;
	const int MAX_KP = keypoints->maxCount;
	
	if(!nonmax_suppression) {
		d_kp = d_kp_final;
	}
	
	// EXECUTE KERNEL
	MEASURE(m, "GPU", "computation", "FAST", "02_03_FAST_kp_GPU", level,
		calcKeypoints_gpu(d_img, img.cols, img.rows, img.step, d_kp, MAX_KP, d_score, threshold, d_counter, stream, computationFinished);
	)
	if(nonmax_suppression) {
		MEASURE(m, "GPU", "computation", "FAST", "02_04_FAST_nonmax_GPU", level,
			nonmaxSuppression_gpu(img.step, d_kp, d_counter, d_score, d_kp_final, d_response, d_count, stream, computationFinished);
		)
	} else {
		//CUDA_SAFE_CALL( cudaMemcpyAsync(d_count, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream) );
	}
	
	CUDA_SAFE_CALL( cudaMemcpyAsync(&(keypoints->count), d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream) );
	CUDA_SAFE_CALL( cudaEventRecord(keypoints->numCopied, stream) );
	
	//keypoints.finishProcessing();
	return m;
}

/* TODO: this headers here needed by next function. Should be reorganized*/
#include <memory>
#include <iostream>
#include <opencv2/core/utility.hpp>

template<int patternSize>
int cornerScore(const uchar* ptr, const int pixel[], int threshold);

template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    //assert((n & (n - 1)) == 0); // n is a power of 2
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

void makeOffsets(int pixel[25], int rowStride, int patternSize)
{
    static const int offsets16[][2] =
    {
        {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
        {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
    };

    static const int offsets12[][2] =
    {
        {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
        {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
    };

    static const int offsets8[][2] =
    {
        {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
        {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
    };

    const int (*offsets)[2] = patternSize == 16 ? offsets16 :
                              patternSize == 12 ? offsets12 :
                              patternSize == 8  ? offsets8  : 0;

    CV_Assert(pixel && offsets);

    int k = 0;
    for( ; k < patternSize; k++ )
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
    for( ; k < 25; k++ )
        pixel[k] = pixel[k - patternSize];
}

template<>
int cornerScore<16>(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 8, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);

    {
        int a0 = threshold;
        for( k = 0; k < 16; k += 2 )
        {
            int a = std::min((int)d[k+1], (int)d[k+2]);
            a = std::min(a, (int)d[k+3]);
            if( a <= a0 )
                continue;
            a = std::min(a, (int)d[k+4]);
            a = std::min(a, (int)d[k+5]);
            a = std::min(a, (int)d[k+6]);
            a = std::min(a, (int)d[k+7]);
            a = std::min(a, (int)d[k+8]);
            a0 = std::max(a0, std::min(a, (int)d[k]));
            a0 = std::max(a0, std::min(a, (int)d[k+9]));
        }

        int b0 = -a0;
        for( k = 0; k < 16; k += 2 )
        {
            int b = std::max((int)d[k+1], (int)d[k+2]);
            b = std::max(b, (int)d[k+3]);
            b = std::max(b, (int)d[k+4]);
            b = std::max(b, (int)d[k+5]);
            if( b >= b0 )
                continue;
            b = std::max(b, (int)d[k+6]);
            b = std::max(b, (int)d[k+7]);
            b = std::max(b, (int)d[k+8]);

            b0 = std::min(b0, std::max(b, (int)d[k]));
            b0 = std::min(b0, std::max(b, (int)d[k+9]));
        }

        threshold = -b0 - 1;
    }
    
    return threshold;
}

template<int patternSize>
void FAST_tt(uchar* imgData, int cols, int rows, int step, CustomVector<cv::KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    const int K = patternSize/2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, step, patternSize);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
        threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    std::unique_ptr<uchar[]> _buf(new uchar[(cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128]);
    uchar* buf[3];
    buf[0] = &_buf[0]; buf[1] = buf[0] + cols; buf[2] = buf[1] + cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + cols + 1;
    cpbuf[2] = cpbuf[1] + cols + 1;
    memset(buf[0], 0, cols*3);

    uchar* ptr = imgData + 3*step + 3;
    for(i = 3; i < rows-2; i++)
    {
		uchar* original_ptr = ptr;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, cols);
        int ncorners = 0;

        if( i < rows - 3 )
        {
            j = 3;
            for( ; j < cols - 3; j++, ptr++ )
            {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }

                if( d & 2 )
                {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                    {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                        {
                            if( ++count > K )
                            {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                    curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                            }
                        }
                        else
                            count = 0;
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

		ptr = original_ptr + step;
		
        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
               (score > prev[j+1] && score > prev[j-1] &&
                score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.push_back(cv::KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
}

MeasureReturn FAST_kernel::execute_cpu() {
	MeasureReturn m(currentFrameNumber);
	cv::Mat img = src->img;
	MEASURE(m, "CPU", "computation", "FAST", "02_03_FAST_kp_and_nonmax_CPU", level,
		FAST_tt<16>(static_cast<uchar*>(img.data), img.cols, img.rows, img.step, *keypoints, threshold, nonmax_suppression);
    )/*
    if(level == 1) {
		std::vector<cv::KeyPoint> k;
		FAST(img,k,threshold,true);
        cv::Mat cv1;
        cv::drawKeypoints(img, k, cv1);
        ORB_SLAM2::Viewer::viewer->push_back("FAST", cv1);
        
        cv::Mat cv2;
        cv::drawKeypoints(img, std::vector<cv::KeyPoint>(keypoints->vec, keypoints->vec+keypoints->count), cv2);
        ORB_SLAM2::Viewer::viewer->push_back("FAST CUSTOM", cv2);
	}*/
    
    //std::cout << "FAST " << level << " - sz: " << img.cols << "x" << img.rows << std::endl;
    //std::cout << "FAST " << level << " - sz: " << keypoints->size() << std::endl;
    //keypoints.finishProcessing();
    return m;
}


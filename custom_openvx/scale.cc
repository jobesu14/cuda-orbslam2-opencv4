#include "functions.hpp"
#include "common.hpp"

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudawarping.hpp>

void Scale::print_size_input() {
	std::cout << src->size_element() << std::endl;
}
void Scale::wait_output() {
	dst->waitTransfer();
}

void Scale::transfer_input() {
	src->wait(where);
}
void Scale::transfer_output() {
	dst->setWhere(where);
	dst->finishProcessing();
}
void Scale::shouldTransferOutput( bool b ) {
	dst->setToBeTransferred( b );
}

MeasureReturn Scale::execute_cpu() {
	MeasureReturn m(currentFrameNumber);
	
	MEASURE(m, "CPU", "computation", "SCALE", "01_01_SCALE_execute_CPU", level,
    /*int old_cols = src->cols;
    int old_rows = src->rows;
    int new_cols = dst->cols;
    int new_rows = dst->rows;
    
    float invX = static_cast<float>(old_cols) / static_cast<float>(new_cols);
    float invY = static_cast<float>(old_rows) / static_cast<float>(new_rows);
    
    uchar *old_image = src->img.data;
    uchar *new_image = dst->img.data;
       
    int idx = 0;
    for(int y = 0; y < new_rows; ++y) {
		for(int x = 0; x < new_cols; ++x) {
            float old_x = std::max(std::min(static_cast<float>(x) * invX, static_cast<float>(old_cols)), 0.0f);
            float old_y = std::max(std::min(static_cast<float>(y) * invY, static_cast<float>(old_rows)), 0.0f);
            
            int old_min_x = static_cast<int>(std::floor(old_x));
            int old_max_x = static_cast<int>(std::ceil(old_x));
            int old_min_y = static_cast<int>(std::floor(old_y));
            int old_max_y = static_cast<int>(std::ceil(old_y));
            old_min_x = std::max(old_min_x, 0);
            old_min_y = std::max(old_min_y, 0);
            old_max_x = std::min(old_max_x, old_cols);
            old_max_y = std::min(old_max_y, old_rows);
            
            float upper_x;
            float lower_x;
            if(old_min_x == old_max_x) {
                upper_x = old_image[old_min_y*old_cols + old_min_x];
                lower_x = old_image[old_max_y*old_cols + old_min_x];
            } else {
                upper_x = (1 - (old_x - old_min_x)) * old_image[old_min_y*old_cols + old_min_x] + 
                          (1 - (old_max_x - old_x)) * old_image[old_min_y*old_cols + old_max_x];
                lower_x = (1 - (old_x - old_min_x)) * old_image[old_max_y*old_cols + old_min_x] + 
                          (1 - (old_max_x - old_x)) * old_image[old_max_y*old_cols + old_max_x];
            }
            
            if(old_min_y == old_max_y) {
                new_image[idx++] = lower_x;
            } else {
                new_image[idx++] = (1 - (old_y - old_min_y)) * upper_x + 
                                   (1 - (old_max_y - old_y)) * lower_x;
            }
        }
    }*/
    cv::Mat imgsrc = src->img;
	cv::Mat imgdst = dst->img;
	
    cv::resize(imgsrc, imgdst, imgdst.size(), 0, 0, cv::INTER_LINEAR);
    )
    
	return m;
}

int scale_gpu(uchar* image_to_scale, int cols, int rows, int step, uchar* image_new, int cols_new, int rows_new, int step_new, float invX, float invY, cudaStream_t stream, cudaEvent_t computationFinished);
MeasureReturn Scale::execute_gpu() {
	MeasureReturn m(currentFrameNumber);
	
	cv::cuda::GpuMat imgsrc = src->imgDevice;
	cv::cuda::GpuMat imgdst = dst->imgDevice;
	uchar* d_img = src->d_img;
	uchar* d_img_scaled = dst->d_img;
	
	int old_cols = src->cols;
    int old_rows = src->rows;
    int new_cols = dst->cols;
    int new_rows = dst->rows;
    
    float invX = static_cast<float>(old_cols) / static_cast<float>(new_cols);
    float invY = static_cast<float>(old_rows) / static_cast<float>(new_rows);
    cv::cuda::Stream stream2 = cv::cuda::StreamAccessor::wrapStream(stream);
	for(int i = 0; i < 7; i++) {
		MEASURE(m, "GPU", "computation", "SCALE", "01_03_SCALE_execute_GPU", level,
		//scale_gpu(d_img, src->cols, src->rows, src->step, d_img_scaled, dst->cols, dst->rows, dst->step, invX, invY, stream, computationFinished);
		cv::cuda::resize(imgsrc, imgdst, imgdst.size(), 0, 0, cv::INTER_LINEAR, stream2);
		)
	}
	
	return m;
}

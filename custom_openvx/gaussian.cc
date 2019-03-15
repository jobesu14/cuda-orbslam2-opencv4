#include "functions.hpp"
#include "common.hpp"

#include <opencv2/core/cuda_stream_accessor.hpp>
void Gaussian::print_size_input() {
	std::cout << src->size_element() << std::endl;
}
void Gaussian::shouldTransferOutput( bool b ) {
	dst->setToBeTransferred( b );
}

MeasureReturn Gaussian::execute_cpu() {
	MeasureReturn m(currentFrameNumber);
	
	MEASURE(m, "CPU", "computation", "GAUSSIAN", "01_01_SCALE_execute_CPU", level,
	cv::Mat imgsrc = src->img;
	cv::Mat imgdst = dst->img;
	/*uchar* data = imgsrc.data;
	uchar* datadst = imgdst.data;
	
	float k[7];
	k[0] = 0.071303f;
	k[1] = 0.131514f;
	k[2] = 0.189879f;
	k[3] = 0.214607f;
	k[4] = 0.189879f;
	k[5] = 0.131514f;
	k[6] = 0.071303f;
	
	
	for(int r = 0; r < imgsrc.rows; ++r) {
		for(int c = 0; c < imgsrc.cols; ++c) {
			float v = 0;
			for(int l = -3; l <= 3; ++l) {
				int new_c = c + l;
				if(new_c >= 0 && new_c < imgsrc.cols) {
					v += k[l+3]*data[r*imgsrc.step+new_c];
				}
			}
			datadst[r*imgsrc.step + c] = v;
		}
	}
	
	for(int r = 0; r < imgsrc.rows; ++r) {
		for(int c = 0; c < imgsrc.cols; ++c) {
			float v = 0;
			for(int l = -3; l <= 3; ++l) {
				int new_r = r + l;
				if(new_r >= 0 && new_r < imgsrc.rows) {
					v += k[l+3]*datadst[new_r*imgsrc.step+c];
				}
			}
			datadst[r*imgsrc.step + c] = v;
		}
	}*/
	cv::GaussianBlur(imgsrc, imgdst, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
	);
	
	return m;
}

int gaussian_gpu(uchar* image_to_gauss, int cols, int rows, int step, uchar* image_new, cudaStream_t stream, cudaEvent_t computationFinished);
MeasureReturn Gaussian::execute_gpu() {
	MeasureReturn m(currentFrameNumber);
	
	cv::cuda::GpuMat imgsrc = src->imgDevice;
	cv::cuda::GpuMat imgdst = dst->imgDevice;
	uchar* d_img = src->d_img;
	uchar* d_img_scaled = dst->d_img;
	
	cv::cuda::Stream stream2 = cv::cuda::StreamAccessor::wrapStream(stream);
	MEASURE(m, "GPU", "computation", "GAUSSIAN", "01_03_SCALE_execute_GPU", level,
		//gaussian_gpu(d_img, src->cols, src->rows, src->step, d_img_scaled, stream, computationFinished);
		filter->apply(imgsrc, imgdst, stream2);
	)
	
	return m;
}

void Gaussian::transfer_input() {
	src->wait(where);
}
void Gaussian::transfer_output() {	
	dst->setWhere(where);
	dst->finishProcessing();
}

void Gaussian::wait_output() {	
	dst->waitTransfer();
}

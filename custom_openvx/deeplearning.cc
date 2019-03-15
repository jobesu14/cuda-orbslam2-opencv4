#include "functions.hpp"
#include "common.hpp"

void DeepLearning::print_size_input() {
	std::cout << src->size_element() << std::endl;
}
void DeepLearning::shouldTransferOutput( bool b ) {
	dst.setToBeTransferred( b );
}

MeasureReturn DeepLearning::execute_cpu() {
	MeasureReturn m(currentFrameNumber);
	
	MEASURE(m, "CPU", "computation", "GAUSSIAN", "01_01_SCALE_execute_CPU", level,
	cv::Mat imgsrc = src->img;
	cv::Mat dstsrc = dst.img;
	uchar* data = imgsrc.data;
	uchar* datadst = dstsrc.data;
	
	float k[5];
	k[0] = 0.0625f;
	k[1] = 0.25f;
	k[2] = 0.375f;
	k[3] = 0.25f;
	k[4] = 0.0625f;
	
	for(int r = 0; r < imgsrc.rows; ++r) {
		for(int c = 0; c < imgsrc.cols; ++c) {
			float v = 0;
			for(int l = -2; l < 2; ++l) {
				int new_c = c + l;
				if(new_c >= 0 && new_c < imgsrc.cols) {
					v += k[l+2]*data[r*imgsrc.step+new_c];
				}
			}
			datadst[r*imgsrc.step + c] = v;
		}
	}
	
	for(int r = 0; r < imgsrc.rows; ++r) {
		for(int c = 0; c < imgsrc.cols; ++c) {
			float v = 0;
			for(int l = -2; l < 2; ++l) {
				int new_r = r + l;
				if(new_r >= 0 && new_r < imgsrc.rows) {
					v += k[l+2]*datadst[new_r*imgsrc.step+c];
				}
			}
			datadst[r*imgsrc.step + c] = v;
		}
	}
	);
	
	return m;
}

int gaussian_gpu(uchar* image_to_gauss, int cols, int rows, int step, uchar* image_new, cudaStream_t stream, cudaEvent_t computationFinished);
MeasureReturn DeepLearning::execute_gpu() {
	MeasureReturn m(currentFrameNumber);
	
	for(int i = 0; i < time_us/500; i++) {
		uchar* d_img = src->d_img;
		uchar* d_img_scaled = dst.d_img;

		//int old_cols = src->cols;
		//int old_rows = src->rows;
		//int new_cols = dst->cols;
		//int new_rows = dst->rows;

		MEASURE(m, "GPU", "computation", "GAUSSIAN", "01_03_SCALE_execute_GPU", level,
			gaussian_gpu(d_img, src->cols, src->rows, src->step, d_img_scaled, stream, computationFinished);
		)
	}
	
	return m;
}

void DeepLearning::transfer_input() {
	src->wait(where);
}
void DeepLearning::transfer_output() {	
	dst.setWhere(where);
	dst.finishProcessing();
}

void DeepLearning::wait_output() {	
	dst.waitTransfer();
}

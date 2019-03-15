#include "data_types.hpp"
#include "params.hpp"

#include <opencv2/cudafilters.hpp>

class Kernel {
public:
	int currentFrameNumber;
	ProcessingElement where;
	int level;
	std::string name;
	cudaStream_t stream;
	cudaEvent_t computationFinished;
	int cid;
	
	Kernel() {
		cudaEventCreate(&computationFinished);
		cid = 0;
		
		level = -1;
		currentFrameNumber = -1;
	}
	~Kernel() {
		cudaEventDestroy(computationFinished);
	}
	
	virtual void transfer_input() = 0;
	virtual void transfer_output() = 0;
	virtual MeasureReturn execute_cpu() = 0;
	virtual MeasureReturn execute_gpu() = 0;
	virtual void wait_output() = 0;
	virtual void print_size_input() = 0;
	virtual void shouldTransferOutput( bool b ) = 0;
	virtual void setWhere(ProcessingElement e) = 0;
	
	void wait() {
		if(where == GPU) {
			//TODO: enable for profiling
			CUDA_SAFE_CALL( cudaEventSynchronize(computationFinished) );
		}
	}
	virtual MeasureReturn execute() {
		MeasureReturn m = (where == GPU) ? execute_gpu() : execute_cpu();
		if(where == GPU)
			CUDA_SAFE_CALL(cudaEventRecord(computationFinished, stream) );
		//transfer_output();
		return m;
	}
};

class Scale : public Kernel {
public:
	Image* src;
	Image* dst;
	
	Scale(Image* src, Image* dst, ProcessingElement where) : Kernel() {
		this->src = src;
		this->dst = dst;
		this->where = where;
		
		src->increaseOut();
		this->cid = 0;
	}
	
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = e; }
	//virtual MeasureReturn execute();
};

class FAST_kernel : public Kernel {
public:
	Image* src;
	ImageI* score;
	CustomVector<cv::KeyPoint>* keypoints;
	CustomVector<cv::KeyPoint> keypoints_tmp;
	int threshold;
	bool nonmax_suppression;
	
	unsigned int* d_counter;
	unsigned int* d_count;
	
	FAST_kernel(Image* src, ImageI* score, CustomVector<cv::KeyPoint>* keypoints, int threshold, bool nonmaxSuppression, ProcessingElement where) : Kernel(), keypoints_tmp(keypoints->maxCount) {
		this->src = src;
		this->score = score;
		this->keypoints = keypoints;
		this->where = where;
		this->nonmax_suppression = nonmaxSuppression;
		this->threshold = threshold;
		
		src->increaseOut();
		this->cid = 1;
		
		CUDA_SAFE_CALL(cudaMalloc(&d_counter, sizeof(unsigned int)));
		if(this->nonmax_suppression) {
			CUDA_SAFE_CALL(cudaMalloc(&d_count, sizeof(unsigned int)));
		} else {
			d_count = d_counter;
		}
	}
	
	~FAST_kernel() {
		CUDA_SAFE_CALL(cudaFree(d_counter));
		if(this->nonmax_suppression) {
			CUDA_SAFE_CALL(cudaFree(d_count));
		}
	}
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = e; }
};

class Gaussian : public Kernel {
public:
	Image* src;
	Image* dst;
	cv::Ptr<cv::cuda::Filter> filter;
	
	Gaussian(Image* src, Image* dst, ProcessingElement where) : Kernel() {
		this->src = src;
		this->dst = dst;
		this->where = where;
		
		src->increaseOut();
		this->cid = 2;
		
		filter = cv::cuda::createGaussianFilter(dst->imgDevice.type(),
					dst->imgDevice.type(), cv::Size(7,7), 2, 2);
	}
	
	~Gaussian() {
	}
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = e; }
};

class ComputeGrid : public Kernel {
public:
	CustomVector<cv::KeyPoint>* src;
	CustomVector<cv::KeyPoint>* dst;
	int threshold;
	int min_y;
	int min_x;
	int max_x;
	int max_y;
	int width;
	
	
	ComputeGrid(CustomVector<cv::KeyPoint>* src, CustomVector<cv::KeyPoint>* dst, int min_x, int min_y, int max_x, int max_y, int threshold, ProcessingElement where) : Kernel() {
		this->src = src;
		this->dst = dst;
		this->threshold = threshold;
		this->where = where;
		
		this->min_x = min_x;
		this->max_x = max_x;
		this->min_y = min_y;
		this->max_y = max_y;
		
		src->increaseOut();
		this->cid = 3;
	}
	
	~ComputeGrid() {
	}
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = CPU; }
};


class ComputeQuadtree : public Kernel {
public:
	CustomVector<cv::KeyPoint>* src;
	CustomVector<cv::KeyPoint>* dst;
	int threshold;
	int min_y;
	int min_x;
	int max_x;
	int max_y;
	int numOfFeatures;
	int width;
	
	
	ComputeQuadtree(CustomVector<cv::KeyPoint>* src, CustomVector<cv::KeyPoint>* dst, int width, int min_x, int min_y, int max_x, int max_y, int numOfFeatures, ProcessingElement where) : Kernel() {
		this->src = src;
		this->dst = dst;
		this->where = where;
		this->width = width;
				
		this->numOfFeatures = numOfFeatures;
		
		this->min_x = min_x;
		this->max_x = max_x;
		this->min_y = min_y;
		this->max_y = max_y;
		
		src->increaseOut();
		this->cid = 4;
	}
	
	~ComputeQuadtree() {
	}
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = CPU; }
};

class ComputeAngle : public Kernel {
public:
	Image* image;
	CustomVector<cv::KeyPoint>* in_kp;
	CustomVector<cv::KeyPoint>* out_kp;
	std::vector<int> umax;
	
	ComputeAngle(Image* image, CustomVector<cv::KeyPoint>* in_kp, CustomVector<cv::KeyPoint>* out_kp, std::vector<int> umax, ProcessingElement where) : Kernel() {
		this->where = where;
		
		this->image = image;
		this->in_kp = in_kp;
		this->out_kp = out_kp;
		this->umax = umax;
		
		image->increaseOut();
		in_kp->increaseOut();
		this->cid = 5;
	}
	
	~ComputeAngle() {
	}
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = CPU; }
};

class ORB : public Kernel {
public:
	Image* image;
	CustomVector<cv::KeyPoint>* in_kp;
	Image* out_descriptor;//TODO: better specification...
	
	ORB(Image* image, CustomVector<cv::KeyPoint>* in_kp, Image* out_descriptor, ProcessingElement where) : Kernel() {
		this->where = where;
		
		this->image = image;
		this->in_kp = in_kp;
		this->out_descriptor = out_descriptor;
		
		image->increaseOut();
		in_kp->increaseOut();
		this->cid = 6;
	}
	
	~ORB() {
	}
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = CPU; }
};

class ScaleCustomVector : public Kernel {
public:
	CustomVector<cv::KeyPoint>* in_kp;
	CustomVector<cv::KeyPoint>* out_kp;
	float scale;
	float scalePatch;
	
	ScaleCustomVector(CustomVector<cv::KeyPoint>* in_kp, CustomVector<cv::KeyPoint>* out_kp, float scale, float scalePatch, ProcessingElement where) : Kernel() {
		this->where = where;
		
		this->in_kp = in_kp;
		this->out_kp = out_kp;
		this->scale = scale;
		this->scalePatch = scalePatch;
		
		in_kp->increaseOut();
		this->cid = 7;
	}
	
	~ScaleCustomVector() {
	}
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );
	
	void setWhere(ProcessingElement e) { this->where = CPU; }
};

#include <unistd.h>

class DeepLearning : public Kernel {
public:
	Image* src;
	//Image dst;
	long time_us;
	
	DeepLearning(Image* src, long time_us) : Kernel()/*, dst(2000, 2000) */{
		this->where = GPU;
		
		this->src = src;
		this->time_us = time_us;
		
		src->increaseOut();
		this->cid = 8;
	}
	
	~DeepLearning() {
	}
	
	void setTime( long time_us ) { this->time_us = time_us; }
	
	//virtual MeasureReturn execute();
	MeasureReturn execute_cpu() { usleep(time_us); return MeasureReturn(-1); }
	MeasureReturn execute_gpu() { usleep(time_us); return MeasureReturn(-1); }
	void transfer_input() { src->wait(where); }
	void transfer_output() { }
	void wait_output() { }
	void print_size_input() { std::cout << src->size_element() << std::endl; }
	void shouldTransferOutput( bool b ) { }
	/*MeasureReturn execute_cpu();
	MeasureReturn execute_gpu();
	void transfer_input();
	void transfer_output();
	void wait_output();
	void print_size_input();
	void shouldTransferOutput( bool b );*/
	
	void setWhere(ProcessingElement e) { this->where = GPU; }
};

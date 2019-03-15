#pragma once
#include "common.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <mutex>
#include <opencv2/core/cuda.hpp>

struct Collector {
	std::vector<std::string> v;
	std::vector<std::string> v_old;
	std::mutex m;
	Collector() {}
	void push_back(std::string s) { 
		std::lock_guard<std::mutex> l(m); 
		v.push_back(s); 
	}
};

#include "semaphore.hpp"
extern Collector c;

enum ProcessingElement {
		CPU,
		GPU
	};
	
void CUDART_CB notify(void* data);
extern std::mutex mutex_data;
struct ProcessingObject {
	ProcessingElement where;
	int numElementOutAttached;
	
	bool onGPU;
	bool onCPU;
	cudaStream_t streamMemory;
	cudaEvent_t objectCopied;
	bool needsToBeTransferred;
	bool transferring;
	std::string name;
	std::mutex m;
	
	int level;
	Semaphore s;
	ProcessingObject() {
		onGPU = false;
		onCPU = false;
		transferring = false;
		numElementOutAttached = 0;
		
		needsToBeTransferred = true;
		
		level = -1;
		
		streamMemory = 0; //default stream
		cudaEventCreateWithFlags(&objectCopied, cudaEventDisableTiming);
	}
	
	~ProcessingObject() {
		cudaEventDestroy(objectCopied);
	}
	
	virtual void toGPU() = 0;
	virtual void toCPU() = 0;
	virtual void finalize_transfer() = 0;
	
	void setToBeTransferred(bool b) {
		needsToBeTransferred = b;
	}
	
	void setDirt() {
		//setWhere(where);
		onCPU = false;
		onGPU = false;
	}
	
	void setWhere(ProcessingElement e) {
		where = e;
		onCPU = false;
		onGPU = false;
		if(e == GPU) onGPU = true;
		if(e == CPU) onCPU = true;
	}
	
	void increaseOut() {
		numElementOutAttached++;
	}
	
	void notifyAll() {
		//std::cerr << "Level " << this->level << " - " << this->name << " releasing x " << numElementOutAttached<< std::endl << std::flush;
		c.push_back("Level " + std::to_string(level) + " -        " + name + " releasing x " + std::to_string(numElementOutAttached));
		for(int i = 0; i < numElementOutAttached; ++i) {
			s.notify();
		}
	}

	void finishProcessing()
	{
		setWhere(where);
		
		{
			std::lock_guard<std::mutex> l(m);
			if(needsToBeTransferred && ! transferring) {
				// issue transfer command
				if(where == GPU) toCPU();
				if(where == CPU) toGPU();
				//serve piu
				CUDA_SAFE_CALL( cudaEventRecord(objectCopied, streamMemory) );
				transferring = true;
				
				// callback non servono perche cudaEvent
				//cudaLaunchHostFunc( stream, notify, this);
				//cudaMemcpyAsync.......
				//cudaStreamAddCallback(stream, 
			}
		}
		
		notifyAll();
	}
	
	void wait() {
		//std::cerr << "Level " << this->level << " - " << this->name << " acquiring x 1" << std::endl << std::flush;
		c.push_back("Level " + std::to_string(level) + " -        " + name + " acquiring x 1");
		s.wait();
		waitTransfer();
	}
	
	void wait(ProcessingElement destination) {
		/*{std::lock_guard<std::mutex> lm(mutex_data);
		std::cout << "Waiting " << name << "-" << level << std::endl;
	}*/
		c.push_back("Level " + std::to_string(level) + " -        " + name + " acquiring v2 x 1");
		//std::cerr << "Level " << this->level << " - " << this->name << " acquiring x 1 v2" << std::endl << std::flush;
		s.wait();
		/*{std::lock_guard<std::mutex> lm(mutex_data);
		std::cout << "Waiting " << name << "-" << level << " DONE" << std::endl;
	}*/
		//if(destination != where) {
			// wait the transfer to be complete
			waitTransfer();
		//}
	}
	
	void waitTransfer() {
		if(transferring) {
			CUDA_SAFE_CALL(cudaEventSynchronize(objectCopied));
			
			std::unique_lock<std::mutex> l(m);
			
			if(transferring)
				finalize_transfer();
			transferring = false;
			onCPU = true;
			onGPU = true;
			//printf("Image %d finished transferring\n", level);
		}
	}
	
	virtual std::string size_element() = 0;
};
/*
void CUDART_CB notify(void* data) {
	ProcessingObject* po;
	
	po->notifyAll();
}*/

template<typename T, int type>
struct Image_ : ProcessingObject{
	int cols;
	int rows;
	int step;
	cv::Mat img;
	cv::cuda::GpuMat imgDevice;
	T* d_img;
	
	Image_() : ProcessingObject(), d_img(nullptr){
	}
	
	Image_(int cols, int rows) : ProcessingObject(), d_img(nullptr){
		std::cout << "Size of T: " << sizeof(T) << std::endl << std::flush;
		init(cols, rows);
	}
	
	//TODO: rule of five
	/*Image_(const Image_& other) { // move constructor
		
	}*/
	
	~Image_() {
		deinit();
	}
	
	void init(int cols, int rows) {
		deinit();
		
		img = cv::Mat::zeros(cv::Size(cols, rows), type);
		CUDA_SAFE_CALL( cudaMalloc( &d_img, cols*rows*sizeof(T) ) );
		imgDevice = cv::cuda::GpuMat( rows, cols, type, d_img);
		
		this->cols = img.cols;
		this->rows = img.rows;
		this->step = img.step;
	}
	
	void deinit() {
		if( d_img != nullptr ) {
			cudaFree(d_img);
		}
	}
	
	void toGPU() {
		//std::unique_lock<std::mutex> l(m);
		
		if(where == CPU && !onGPU && !transferring) {
			//std::cout << "To GPU" << std::endl;
			CUDA_SAFE_CALL(cudaMemcpyAsync(d_img, img.ptr<T>(0), sizeof(T)*rows*img.cols, cudaMemcpyHostToDevice, streamMemory));
			transferring = true;
			//std::cout << "To GPU done!" << std::endl;
		}
	}
	
	void toCPU() {
		//std::unique_lock<std::mutex> l(m);
		
		if(where == GPU && !onCPU) {
			//std::cout << "To CPU" << std::endl;
			CUDA_SAFE_CALL(cudaMemcpyAsync(img.ptr<T>(0), d_img, sizeof(T)*rows*img.cols, cudaMemcpyDeviceToHost, streamMemory));
			transferring = true;
			//std::cout << "To CPU done" << std::endl;
		}
	}
	
	void finalize_transfer() {
		//already did everyting!
	}
	
	void setRows(int r) {
		this->rows = r;
	}
	std::string size_element() { return std::to_string(cols) + "x" + std::to_string(rows); }
};


using Image = Image_<uchar, CV_8UC1>;
using ImageI = Image_<int, CV_32SC1>;

template<typename T>
struct CustomVector : ProcessingObject {
	T* vec;
	T* d_vec;
	int count;
	unsigned int *d_count;
	int maxCount;
	cudaEvent_t numCopied;
	
	/*CustomVector() : ProcessingObject(), this(8) {
	}*/
	CustomVector(int maxCount) : ProcessingObject() {
		vec = new T[maxCount];
		CUDA_SAFE_CALL( cudaMalloc( &d_vec, maxCount*sizeof(T) ) );
		CUDA_SAFE_CALL( cudaMalloc( &d_count, sizeof(unsigned int) ) );
		count = 0;
		this->maxCount = maxCount;
		where = CPU;
		numElementOutAttached = 0;
		
		CUDA_SAFE_CALL( cudaEventCreateWithFlags(&numCopied, cudaEventDisableTiming) );
	}
		
	~CustomVector() {
		CUDA_SAFE_CALL( cudaFree( d_vec ) );
		CUDA_SAFE_CALL( cudaFree( d_count ) );
		CUDA_SAFE_CALL( cudaEventDestroy(numCopied) );
		delete[] vec;
	}
	
	void toGPU() {
		if(where == CPU) {
			CUDA_SAFE_CALL(cudaMemcpyAsync(d_vec, vec, sizeof(T)*count, cudaMemcpyHostToDevice, streamMemory));
		}
	}
	
	void setMaxCount(int maxCount) {
		this->maxCount = maxCount;
	}
	
	void toCPU() {
		if(where == GPU) {
			// MUST BE SYNCHRONOUS
			CUDA_SAFE_CALL( cudaEventSynchronize( numCopied ) );
			CUDA_SAFE_CALL(cudaMemcpyAsync(vec, d_vec, sizeof(T)*count, cudaMemcpyDeviceToHost, streamMemory));
		}
	}
	void finalize_transfer() {
		//already did everyting!
	}
	
	void setSize(int count) {
		this->count = count;
	}
	
	std::string size_element() { return std::to_string(count); }
};


template<>
struct CustomVector<cv::KeyPoint> : ProcessingObject {
	cv::KeyPoint* vec;
	CustomVector<short2> kp;
	short2* h_kp;
	short2* d_kp;
	CustomVector<float> response;
	float* h_response;
	float* d_response;
	int count;
	int *d_count;
	int maxCount;
	cudaEvent_t numCopied;
	
	CustomVector() : ProcessingObject(), kp(70000), response(70000) {
		count = 0;
		this->maxCount = 70000;
		where = CPU;
		numElementOutAttached = 0;
		
		d_kp = kp.d_vec;
		d_response = response.d_vec;
		
		h_kp = kp.vec;
		h_response = response.vec;
		init(maxCount);
		
		CUDA_SAFE_CALL( cudaEventCreateWithFlags(&numCopied, cudaEventDisableTiming) );
	}
	
	CustomVector(int maxCount) : ProcessingObject(), kp(maxCount), response(maxCount) {
		count = 0;
		this->maxCount = maxCount;
		where = CPU;
		numElementOutAttached = 0;
		
		d_kp = kp.d_vec;
		d_response = response.d_vec;
		
		h_kp = kp.vec;
		h_response = response.vec;
		
		CUDA_SAFE_CALL( cudaEventCreateWithFlags(&numCopied, cudaEventDisableTiming) );
		init(maxCount);
	}
	
	void init(int n) {
		vec = new cv::KeyPoint[maxCount];
	}
	
	~CustomVector() {
		CUDA_SAFE_CALL( cudaEventDestroy(numCopied) );
		delete[] vec;
	}
	
	void toGPU() {
		if(where == CPU) {
			for(int i = 0; i < count; ++i) {
				//cv::KeyPoint kp(kps[i].x, kps[i].y, 7, -1, response[i]);//FEATURE_SIZE
				h_kp[i].x = vec[i].pt.x;
				h_kp[i].y = vec[i].pt.y;
				h_response[i] = vec[i].response;
			}
			CUDA_SAFE_CALL( cudaMemcpyAsync(d_kp, h_kp, count*sizeof(short2), cudaMemcpyHostToDevice, streamMemory) );
			CUDA_SAFE_CALL( cudaMemcpyAsync(d_response, h_response, count*sizeof(float), cudaMemcpyHostToDevice, streamMemory) );
		}
	}
	
	void toCPU() {
		if(where == GPU) {
			CUDA_SAFE_CALL( cudaMemcpyAsync(h_kp, d_kp, count*sizeof(short2), cudaMemcpyDeviceToHost, streamMemory) );
			CUDA_SAFE_CALL( cudaMemcpyAsync(h_response, d_response, count*sizeof(float), cudaMemcpyDeviceToHost, streamMemory) );
		}
	}
	void finalize_transfer() {
		if(where == GPU)
		for(int i = 0; i < count; ++i) {
			//cv::KeyPoint kp(kps[i].x, kps[i].y, 7, -1, response[i]);//FEATURE_SIZE
			vec[i].pt.x = h_kp[i].x;
			vec[i].pt.y = h_kp[i].y;
			vec[i].response = h_response[i];
		}
	}
	
	bool clear() {
		count = 0;
		return true;
	}
	bool push_back(cv::KeyPoint k) {
		if(count < maxCount - 1) {
			h_kp[count].x = k.pt.x;
			h_kp[count].y = k.pt.y;
			h_response[count] = k.response;
			vec[count++] = k;
			return true;
		}
		return false;
	}
	
	void setSize(int count) {
		this->count = count;
	}
	int size() {
		return count;
	}
	
	void setMaxCount(int maxCount) {
		this->maxCount = maxCount;
	}
	
	std::string size_element() { return std::to_string(count); }
	
};

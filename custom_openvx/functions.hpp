#pragma once
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <chrono>
#include <thread>
#include "data_types.hpp"
#include "kernels.hpp"
#include "params.hpp"
#include <fstream>
    
#ifdef OPENVX
#ifdef ORB_CUDA
    const int EDGE_THRESHOLD = 19 + 10; // FIXED (why?)
#else
	const int EDGE_THRESHOLD = 19 + 3; // FIXED: Visionworks FAST doesn't consider corners automatically!
#endif
#else
	const int EDGE_THRESHOLD = 19;
#endif

//const int WINDOW_SIZE = 30;


struct MeasureReturnAll {
	std::vector<MeasureReturn> measure;
	std::string fname;
	std::mutex m;
	
	void add(MeasureReturn m) {
		std::lock_guard<std::mutex> lock(this->m);
		measure.push_back(m);
	}
	
	MeasureReturnAll() {
	}
	MeasureReturnAll(std::string fname) : fname(fname){
	}
	
	void setName(std::string fname) {
		this->fname = fname;
	}
	
	~MeasureReturnAll() {
		std::ofstream outfile(fname);
		outfile << "Frame;Executor;OpType;Node;Level;CompleteName;nanoTime;msTime" << std::endl;
		for(auto mm : measure) {
			for(auto m : mm.measure) {
				outfile << mm.frame_number 
				  << ";" << m.executor
				  << ";" << m.execType
				  << ";" << m.routine
				  << ";" << m.level
				  << ";" << m.type
				  << ";" << m.nanoTime
				  << ";" << m.msTime
				  << std::endl;
			}
		}
	}
};


class ORBFunctions {
public:
    ORBFunctions(int width, int height, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST, int _nfeatures, std::string suffix = "");
    
    // Scale
    MeasureReturn prepareFirstImage(cv::Mat image);
    
    void init(bool global_optimum, bool gpu);
    
    void execute(cv::Mat im, MeasureReturnAll& all);
    
//  Images
	Image images[LEVELS_C];
	Image imagesGauss[LEVELS_C];
	CustomVector<cv::KeyPoint> kp_fast[LEVELS_C];
	CustomVector<cv::KeyPoint> kp_grid[LEVELS_C];
	CustomVector<cv::KeyPoint> kp_quadtree[LEVELS_C];
	CustomVector<cv::KeyPoint> kp_angle[LEVELS_C];
	CustomVector<cv::KeyPoint> kp_final[LEVELS_C];
	Image descriptor[LEVELS_C];
	
	
	Scale* scales[LEVELS_C-1];
	FAST_kernel* fast_k[LEVELS_C];
	Gaussian* gaussian[LEVELS_C];
	ComputeGrid* grid[LEVELS_C];
	ComputeQuadtree* quadtree[LEVELS_C];
	ComputeAngle* angle[LEVELS_C];
	ORB* orb[LEVELS_C];
	ScaleCustomVector* scaleVector[LEVELS_C];
	DeepLearning* deepLearning[1];

//private:
	int currentFrameNumber;
    int width;
    int height;
    float mvScaleFactor[LEVELS_C];
    float mvLevelSigma2[LEVELS_C];
    float mvInvScaleFactor[LEVELS_C];
    float mvInvLevelSigma2[LEVELS_C];
    int mnFeaturesPerLevel[LEVELS_C];
    cv::Point pattern[512];
    int umax[HALF_PATCH_SIZE_C + 1];
    
    
// CUDA-related stuffs
    cudaStream_t stream;
    ImageI d_scores[LEVELS_C];
    
    
    // execution
    typedef std::pair<Kernel*, bool> pk;
    std::vector<pk> QUEUES[6];//GPU, CPU 1..4
    std::vector<std::thread> threads;
    
    void thread_run(int idx);
    
    void elaborateQueue(std::vector<pk>* queue, int idx_thread, MeasureReturn& m, bool print = true);
    
    /*Semaphore ss1;
    Semaphore ss2;
    Semaphore *s1;
    Semaphore *s2;*/
    
    
    volatile bool continueToExecute;
    int fn;
    MeasureReturnAll all;
    std::mutex mutexes[6];
    std::mutex mutexes_main[6];
    const int num_threads = 6;
    int global_cpu = 1;
    
    int nfeatures;
    float scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;
    
};

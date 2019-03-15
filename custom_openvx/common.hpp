#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <signal.h>

#define CONCAT_(x, y) x##y
#define CONCAT(x,y) CONCAT_(x,y)
//#define UNIQUE(base) PP_CAT(base, __LINE__

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if(cudaSuccess != err) {\
        fprintf(stderr, "CUDA error in file @ '%s':%i : %d - %s.\n", \
            __FILE__, __LINE__, err, cudaGetErrorString(err) ); \
        raise(SIGINT); \
    } \
} while(0);
#endif


//exit(EXIT_FAILURE);

#ifndef MEASURE
#define MEASURE(m, executor, execType, routine, type, level, instr) \
    auto CONCAT(tmp_beg_, __LINE__) = std::chrono::steady_clock::now(); \
    instr \
    auto CONCAT(tmp_end_, __LINE__) = std::chrono::steady_clock::now(); \
    m.add(Measure(executor, execType, routine, type, \
    std::chrono::duration<long long, std::nano>(CONCAT(tmp_end_, __LINE__) - CONCAT(tmp_beg_, __LINE__)).count(),\
    std::chrono::duration<double, std::milli>(CONCAT(tmp_end_, __LINE__) - CONCAT(tmp_beg_, __LINE__)).count(), level));
#endif


// TODO: profiling classes
class Measure {
public:
	std::string executor;
	std::string execType;
	std::string routine;
	std::string type;
	long long nanoTime;
	double msTime;
	int level;
	
	Measure(std::string executor, std::string execType, std::string routine, std::string type, long long nanoTime, double msTime, int level = -2) : 
		executor(executor), execType(execType), routine(routine),
		type(type), nanoTime(nanoTime), msTime(msTime), level(level) {
			}
};

class MeasureReturn {
public:
	int frame_number;
	
	std::vector<Measure> measure;
	union ret{
		int int_ret;
	};
	
	MeasureReturn(int fN) : frame_number(fN) {};
	
	void add(Measure m) {
		measure.push_back(m);
	}
	
	void clear() {
		measure.clear();
	}
};

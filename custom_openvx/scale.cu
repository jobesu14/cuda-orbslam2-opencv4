#include <memory>
#include <iostream>
#include "functions.hpp"
#include "common.hpp"

__host__ __device__ __forceinline__ int divUp(int total, int grain) {
	return (total + grain - 1) / grain;
}
/*template <typename T>
__device__ __host__ T access(T* p, int y, int x, int cols) {
	return p[y*cols+x];
}*/

#define access(p, cols, y, x) p[(y)*cols+x]

///////////////////////////////////////////////////////////////////////////
// nonmaxSuppression

__global__ void scale(uchar* image_to_scale, int cols, int rows, int step, uchar* image_new, int cols_new, int rows_new, int step_new, float invX, float invY)
{
	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 110)

	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < cols_new && y < rows_new)
	{
		float old_x = max(min(static_cast<float>(x) * invX, static_cast<float>(cols)), 0.0f);
		float old_y = max(min(static_cast<float>(y) * invY, static_cast<float>(rows)), 0.0f);
		
		int old_min_x = static_cast<int>(floor(old_x));
		int old_max_x = static_cast<int>(ceil(old_x));
		int old_min_y = static_cast<int>(floor(old_y));
		int old_max_y = static_cast<int>(ceil(old_y));
		old_min_x = max(old_min_x, 0);
		old_min_y = max(old_min_y, 0);
		old_max_x = min(old_max_x, cols);
		old_max_y = min(old_max_y, rows);
		
		float upper_x, lower_x;
		if(old_min_x == old_max_x) {
			upper_x = image_to_scale[old_min_y*cols + old_min_x];
			lower_x = image_to_scale[old_max_y*cols + old_min_x];
		} else {
			upper_x = (1 - (old_x - old_min_x)) * image_to_scale[old_min_y*cols + old_min_x] + 
					  (1 - (old_max_x - old_x)) * image_to_scale[old_min_y*cols + old_max_x];
			lower_x = (1 - (old_x - old_min_x)) * image_to_scale[old_max_y*cols + old_min_x] + 
					  (1 - (old_max_x - old_x)) * image_to_scale[old_max_y*cols + old_max_x];
		}
		
		if(old_min_y == old_max_y) {
			image_new[y*step_new+x] = lower_x;
		} else {
			image_new[y*step_new+x] = (1 - (old_y - old_min_y)) * upper_x + 
							          (1 - (old_max_y - old_y)) * lower_x;
		}
	}

	#endif
}

int scale_gpu(uchar* image_to_scale, int cols, int rows, int step, uchar* image_new, int cols_new, int rows_new, int step_new, float invX, float invY, cudaStream_t stream, cudaEvent_t event)
{
	dim3 block(32, 8);

	dim3 grid;
	grid.x = divUp(cols - 6, block.x);
	grid.y = divUp(rows - 6, block.y);

	scale<<<grid, block, 0, stream>>>(image_to_scale, cols, rows, step, image_new, cols_new, rows_new, step_new, invX, invY);
	CUDA_SAFE_CALL( cudaGetLastError() );
	//CUDA_SAFE_CALL( cudaEventRecord(event, stream) );

	//CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

	return 0;
}

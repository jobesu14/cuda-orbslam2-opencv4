#include <memory>
#include <iostream>
#include "functions.hpp"
#include "common.hpp"

__host__ __device__ __forceinline__ int divUp(int total, int grain) {
	return (total + grain - 1) / grain;
}

template<bool vertical>
__global__ void  gaussian(uchar* image_to_gauss, int cols, int rows, int step, uchar* image_new)
{
	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 110)

	const int c = threadIdx.x + blockIdx.x * blockDim.x;
	const int r = threadIdx.y + blockIdx.y * blockDim.y;

	//float k[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
	//float k[5] = {1,4,6,4,1};
	float k[7] = {0.071303f, 0.131514f, 0.189879f, 0.214607f, 0.189879f, 0.131514f, 0.071303f};
	
	if (r < rows && c < cols)
	{
		float v = 0;
		if(vertical) {
			for(int l = -3; l <= 3; ++l) {
				int new_r = r + l;
				float value = 0;
				
				if(new_r >= 0 && new_r < rows) {
					value = image_to_gauss[new_r*step+c];
				}else if(new_r < 0) {
					value = image_to_gauss[0*step+c];
				}else {
					value = image_to_gauss[(rows-1)*step+c];
				}
				v += k[l+3]*value;
			}
		} else {
			for(int l = -3; l <= 3; ++l) {
				int new_c = c + l;
				float value = 0;
				
				if(new_c >= 0 && new_c < cols) {
					value = image_to_gauss[r*step+new_c];
				}else if(new_c < 0) {
					value = image_to_gauss[r*step+0];
				}else {
					value = image_to_gauss[r*step+cols-1];;
				}
				
				v += k[l+3]*value;
			}
		}
		image_new[r*step+c] = v;
	}

	#endif
}

int gaussian_gpu(uchar* image_to_gauss, int cols, int rows, int step, uchar* image_new, cudaStream_t stream, cudaEvent_t event)
{
	dim3 block(16, 16);

	dim3 grid;
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);

	gaussian<false><<<grid, block, 0, stream>>>(image_to_gauss, cols, rows, step, image_new);
	gaussian<true><<<grid, block, 0, stream>>>(image_new, cols, rows, step, image_new);
	CUDA_SAFE_CALL( cudaGetLastError() );
	
	//CUDA_SAFE_CALL( cudaEventRecord(event, stream) );

	//CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );

	return 0;
}

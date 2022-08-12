
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void kernel_with_branch(int *_output)
{
	if (threadIdx.x % 2 == 0)
		_output[threadIdx.x] = 1;
	else
		_output[threadIdx.x] = 0;
}

int main()
{
	int isEven[64] = { 0 };
	int* dIsEven = NULL;
	cudaMalloc(&dIsEven, sizeof(int)*64);
	cudaMemset(dIsEven, 0, sizeof(int) * 64);

	kernel_with_branch <<<1, 512>>> (dIsEven);
	
	cudaMemcpy(isEven, dIsEven, sizeof(int) * 64, cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < 64; i++)
		printf("%d ", isEven[i]);
}
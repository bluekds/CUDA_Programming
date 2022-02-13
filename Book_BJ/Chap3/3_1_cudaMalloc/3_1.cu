#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main(void)
{
	int* dDataPtr;
	cudaMalloc(&dDataPtr, sizeof(int) * 32);
}
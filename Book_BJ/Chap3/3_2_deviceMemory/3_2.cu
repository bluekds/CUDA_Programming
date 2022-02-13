#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory(void)
{
	size_t free, total;
	cudaMemGetInfo(&free, &total); // 현재 사용가능한 device memory 크기와, 총 device 메모리 크기를 얻어오는 함수
	printf("Device memory (free/total) = %lld/%lld bytes\n"
		, free, total);
}

int main(void)
{
	int* dDataPtr;
	cudaError_t errorCode ;

	checkDeviceMemory();
	errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024); // device memory 할당
	printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode));
	checkDeviceMemory();

	errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024); // device memory 초기화
	printf("cudaMemset - %s\n", cudaGetErrorName(errorCode));

	errorCode = cudaFree(dDataPtr);
	printf("cudaFree - %s\n", cudaGetErrorName(errorCode)); // device memory 해제
	checkDeviceMemory();
}
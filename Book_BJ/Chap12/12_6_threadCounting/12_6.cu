#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_BLOCK 10240
#define NUM_T_IN_B 512

__global__ void threadCounting_noSync(int* a)
{
	(*a)++;
}

__global__ void threadCounting_atomicGlobal(int* a)
{
	atomicAdd(a, 1);
}

__global__ void threadCounting_atomicShared(int* a)
{
	__shared__ int sa;

	if (threadIdx.x == 0)
		sa = 0;
	__syncthreads();

	atomicAdd(&sa, 1);
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(a, sa);
}

int main(void) {
	DS_timer timer(10);
	timer.setTimerName(0, (char*)"No Sync.");
	timer.setTimerName(1, (char*)"AtomicGlobal");
	timer.setTimerName(2, (char*)"AtomicShared");

	int a = 0;
	int* d1, * d2, * d3;

	//cudaSetDevice(1);

	cudaMalloc((void**)&d1, sizeof(int));
	cudaMemset(d1, 0, sizeof(int) * 0);

	cudaMalloc((void**)&d2, sizeof(int));
	cudaMemset(d2, 0, sizeof(int) * 0);

	cudaMalloc((void**)&d3, sizeof(int));
	cudaMemset(d3, 0, sizeof(int) * 0);

	// warp-up
	threadCounting_noSync << <NUM_BLOCK, NUM_T_IN_B >> > (d1);
	cudaDeviceSynchronize();

	timer.onTimer(0);
	threadCounting_noSync << <NUM_BLOCK, NUM_T_IN_B >> > (d1);
	cudaDeviceSynchronize();
	timer.offTimer(0);

	cudaMemcpy(&a, d1, sizeof(int), cudaMemcpyDeviceToHost);
	printf("[No Sync.] # of threads = %d\n", a);

	timer.onTimer(1);
	threadCounting_atomicGlobal << <NUM_BLOCK, NUM_T_IN_B >> > (d2);
	cudaDeviceSynchronize();
	timer.offTimer(1);

	cudaMemcpy(&a, d2, sizeof(int), cudaMemcpyDeviceToHost);
	printf("[AtomicGlobal] # of threads = %d\n", a);

	timer.onTimer(2);
	threadCounting_atomicShared << <NUM_BLOCK, NUM_T_IN_B >> > (d3);
	cudaDeviceSynchronize();
	timer.offTimer(2);

	cudaMemcpy(&a, d3, sizeof(int), cudaMemcpyDeviceToHost);
	printf("[AtomicShared] # of threads = %d\n", a);

	cudaFree(d1);
	cudaFree(d2);
	cudaFree(d3);

	timer.printTimer();

	return 0;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define RUN_CPU

#define TID_X (threadIdx.x)
#define TID_GLOBAL (blockIdx.x*blockDim.x + threadIdx.x)
#define FIRST_T_IN_BLCOK (threadIdx.x == 0)
#define NUM_T_IN_BLOCK 512

#define LOOP_I(_size) for(int i = 0; i<_size; i++)

__host__ __device__ double f(double x) {
	return x * x;
}

// Atomic operation on global memory
__global__ void trap_kernel_GlobalAtomic(double a, double b, double h, int n, double* sum)
{
	int tid = TID_GLOBAL;

	if (tid >= n - 1)
		return;

	double x_i = a + h * tid;
	double x_j = a + h * (tid + 1);
	double d = (f(x_i) + f(x_j)) / 2.0;

	atomicAdd(sum, d * h);
}

// Ver 1. Atomic operation on shared memory
__global__ void trap_kernel_SharedAtomic(double a, double b, double h, int n, double* sum)
{
	int tid = TID_GLOBAL;
	if (tid >= n - 1)
		return;

	__shared__ double localSum;

	if (FIRST_T_IN_BLCOK)
		localSum = 0;
	__syncthreads();

	double x_i = a + h * tid;
	double x_j = a + h * (tid + 1);
	double d = (f(x_i) + f(x_j)) / 2.0;

	atomicAdd(&localSum, d * h);

	__syncthreads();

	if (FIRST_T_IN_BLCOK)
		atomicAdd(sum, localSum);
}

// Ver 2. Shared memory with Master-slave model
__global__ void trap_kernel_SharedMasterSlave(double a, double b, double h, int n, double* sum)
{
	int tid = TID_GLOBAL;

	__shared__ double localVal[NUM_T_IN_BLOCK];
	localVal[TID_X] = 0;

	if (tid >= n - 1)
		return;

	double x_i = a + h * tid;
	double x_j = a + h * (tid + 1);
	double d = (f(x_i) + f(x_j)) / 2.0;

	localVal[TID_X] = d * h;

	__syncthreads();

	if (FIRST_T_IN_BLCOK) {
		for (int i = 1; i < NUM_T_IN_BLOCK; i++)
			localVal[0] += localVal[i];

		atomicAdd(sum, localVal[0]);
	}
}

// Ver 3. Reduction
__global__ void trap_kernel_Reduction(double a, double b, double h, int n, double* sum)
{
	int tid = TID_GLOBAL;

	__shared__ double localVal[NUM_T_IN_BLOCK];
	localVal[TID_X] = 0;

	if (tid >= n - 1)
		return;

	double x_i = a + h * tid;
	double x_j = a + h * (tid + 1);
	double d = (f(x_i) + f(x_j)) / 2.0;

	localVal[TID_X] = d * h;

	__syncthreads();

	// reduction here
	int offset = 1;

	while (offset < NUM_T_IN_BLOCK) {

		if (threadIdx.x % (2 * offset) == 0)
			localVal[threadIdx.x] += localVal[threadIdx.x + offset];
		__syncthreads();

		offset *= 2;
	}

	if (FIRST_T_IN_BLCOK) {
		atomicAdd(sum, localVal[0]);
	}
}

// Ver 4. Reduction - No bank conflict
__global__ void trap_kernel_ReductionNoBankConflict(double a, double b, double h, int n, double* sum)
{
	int tid = TID_GLOBAL;

	__shared__ double localVal[NUM_T_IN_BLOCK];
	localVal[TID_X] = 0;

	if (tid >= n - 1)
		return;

	double x_i = a + h * tid;
	double x_j = a + h * (tid + 1);
	double d = (f(x_i) + f(x_j)) / 2.0;

	localVal[TID_X] = d * h;

	__syncthreads();

	// reduction here
	int offset = NUM_T_IN_BLOCK / 2;

	while (offset > 0) {
		if (TID_X < offset) {
			localVal[TID_X] += localVal[TID_X + offset];
		}
		offset /= 2;

		__syncthreads();
	}

	if (FIRST_T_IN_BLCOK) {
		atomicAdd(sum, localVal[0]);
	}
}

__device__ void warpReduce(volatile double* _localVal, int _tid)
{
	_localVal[_tid] += _localVal[_tid + 32];
	_localVal[_tid] += _localVal[_tid + 16];
	_localVal[_tid] += _localVal[_tid + 8];
	_localVal[_tid] += _localVal[_tid + 4];
	_localVal[_tid] += _localVal[_tid + 2];
	_localVal[_tid] += _localVal[_tid + 1];
}

// Ver 5. Reduction - Loop unrolling in a warp
__global__ void trap_kernel_LoopUnroll_Warp(double a, double b, double h, int n, double* sum)
{
	int tid = TID_GLOBAL;

	__shared__ double localVal[NUM_T_IN_BLOCK];
	localVal[TID_X] = 0;

	if (tid >= n - 1)
		return;

	double x_i = a + h * tid;
	double x_j = a + h * (tid + 1);
	double d = (f(x_i) + f(x_j)) / 2.0;

	localVal[TID_X] = d * h;

	__syncthreads();

	// reduction here
	int offset = NUM_T_IN_BLOCK / 2;

	while (offset > 32) {
		if (TID_X < offset) {
			localVal[TID_X] += localVal[TID_X + offset];
		}
		offset /= 2;

		__syncthreads();
	}

	if (TID_X < 32)
		warpReduce(localVal, TID_X);

	__syncthreads();

	if (TID_X == 0) {
		atomicAdd(sum, localVal[0]);
	}
}

// Ver 6. Reduction - Loop unrolling in a block
__global__ void trap_kernel_LoopUnroll_Full(double a, double b, double h, int n, double* sum)
{
	int tid = TID_GLOBAL;

	__shared__ double localVal[NUM_T_IN_BLOCK];
	localVal[TID_X] = 0;

	if (tid >= n - 1)
		return;

	double x_i = a + h * tid;
	double x_j = a + h * (tid + 1);
	double d = (f(x_i) + f(x_j)) / 2.0;

	localVal[TID_X] = d * h;

	__syncthreads();

	// reduction here
	if (TID_X < 256)
		localVal[TID_X] += localVal[TID_X + 256];
	__syncthreads();

	if (TID_X < 128)
		localVal[TID_X] += localVal[TID_X + 128];
	__syncthreads();

	if (TID_X < 64)
		localVal[TID_X] += localVal[TID_X + 64];
	__syncthreads();

	if (TID_X < 32)
		warpReduce(localVal, TID_X);

	if (TID_X == 0) {
		atomicAdd(sum, localVal[0]);
	}
}

#define DO_TEST(_name,_timerID) {			\
	timer.onTimer(_timerID);						\
	cudaMemset(dSum, 0, sizeof(double));	\
	trap_kernel_##_name<<<dimGrid, NUM_T_IN_BLOCK>>> (a, b, h, n, dSum); \
	cudaMemcpy(&gpuSum, dSum, sizeof(double), cudaMemcpyDeviceToHost);	\
	timer.offTimer(_timerID);	\
	printf("[GPU] %s = %f\n", #_name, gpuSum);	\
}

void main(void)
{
	double a = -5.0;
	double b = 5.0;
	int n = 1024 * 1024 * 1024;
	double x_i = 0.0; double x_j = 0.0; double h = (b - a) / n; double d = 0.0; double sum = 0.0;

	cudaSetDevice(0);

	DS_timer timer(10);
	timer.setTimerName(0, "[CPU]");
	timer.setTimerName(1, "[Atomic on Global Mem.]");
	timer.setTimerName(2, "[Shared Ver1. Atomic Op. on a variable ]");
	timer.setTimerName(3, "[Shared Ver2. Master-Slave]");

	timer.setTimerName(4, "[Shared Ver3. Reduction]");
	timer.setTimerName(5, "[Shared Ver4. Reduction - Avoid Bank conflict]");
	timer.setTimerName(6, "[Shared Ver5. Reduction - Loop unrolling (warp)]");
	timer.setTimerName(7, "[Shared Ver6. Reduction - Loop unrolling (block)]");

	timer.initTimers();

#ifdef RUN_CPU
	timer.onTimer(0);
	for (int i = 0; i < n - 1; i++) {
		x_i = a + h * i;
		x_j = a + h * (i + 1);
		d = (f(x_i) + f(x_j)) / 2.0;
		sum += d*h;
	}
	timer.offTimer(0);
	printf("CPU = %f\n", sum);
#endif

	double gpuSum = 0;
	double* dSum;
	cudaMalloc(&dSum, sizeof(double));
	cudaMemset(dSum, 0, sizeof(double));

	dim3 dimGrid(ceil(n / (float)NUM_T_IN_BLOCK), 1, 1);

	DO_TEST(GlobalAtomic, 1);
	DO_TEST(SharedAtomic, 2);
	DO_TEST(SharedMasterSlave, 3);
	DO_TEST(Reduction, 4);
	DO_TEST(ReductionNoBankConflict, 5);
	DO_TEST(LoopUnroll_Warp, 6);
	DO_TEST(LoopUnroll_Full, 7);

	cudaFree(dSum);

	timer.printTimer();
}
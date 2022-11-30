﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_CPU_THREADS	(4)

#define ROW_SIZE (32)
#define K_SIZE   (128)
#define COL_SIZE (32)

#define MAT_SIZE_A (ROW_SIZE*K_SIZE)
#define MAT_SIZE_B (K_SIZE*COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE*COL_SIZE)

// input matrix
float A[ROW_SIZE][K_SIZE];	// m * k
float B[K_SIZE][COL_SIZE];	// k * n

// timer
DS_timer* timer;
void setTimer(void);
#define TIMER_HOST			0
#define TIMER_KERNEL		1
#define TIMER_KERNEL_SH		2
#define TIMER_KERNEL_SH_C	3
#define TIMER_HtoD			4
#define TIMER_DtoH			5
#define NUM_TIMER			(TIMER_DtoH+1)

void genInputMatrices(void);

// output matrix
float hostC[ROW_SIZE][COL_SIZE];	// host result
float deviceC[COL_SIZE][COL_SIZE];	// device result

#define memsetZero(_P,_type,_size) memset(_P, 0, sizeof(_type)*_size);
#define dMemAlloc(_P, _type, _size) cudaMalloc(&_P, sizeof(_type)*_size);


__global__ void matMul_kernel(float* _A, float* _B, float* _C)
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	float result = 0;
	for (int k = 0; k < K_SIZE; k++)
		result += _A[row * K_SIZE + k] * _B[col + k * COL_SIZE];
	_C[index] = result;
}

__global__ void matMul_kernel_shared(float* _A, float* _B, float* _C)
{
	int row = threadIdx.x;
	int col = threadIdx.y;
	int index = row * blockDim.y + col;

	__shared__ float sA[ROW_SIZE][K_SIZE];	// 32 * 256 * 4 bytes = 16 KB
	__shared__ float sB[K_SIZE][COL_SIZE];	// 16 KB

	if (row == 0) { // read matrix B
		for (int k = 0; k < K_SIZE; k++)
			sB[k][col] = _B[col + k * COL_SIZE];
	}
	else if (col == 0 ) { // read matrix A
		for (int k = 0; k < K_SIZE; k++)
			sA[row][k] = _A[row * K_SIZE + k];

	}
	if (row == 0 && col == 0) { // read the first row of A
		for (int k = 0; k < K_SIZE; k++)
			sA[row][k] = _A[row * K_SIZE + k];
	}

	__syncthreads(); // wait until all thread load the matrix

	float result = 0;
	for (int k = 0; k < K_SIZE; k++)
		result += sA[row][k] * sB[k][col];
	_C[index] = result;
}

void main(void)
{
	timer = NULL;	setTimer();

	float* dA, * dB, * dC;
	dA = dB = dC = NULL;

	memsetZero(A, float, MAT_SIZE_A);	memsetZero(B, float, MAT_SIZE_B);
	memsetZero(hostC, float, MAT_SIZE_C);	memsetZero(deviceC, float, MAT_SIZE_C);

	// device memory allocaiton
	dMemAlloc(dA, float, MAT_SIZE_A);
	dMemAlloc(dB, float, MAT_SIZE_B);
	dMemAlloc(dC, float, MAT_SIZE_C);

	// generate input matrices
	genInputMatrices();

	// Host code
	timer->onTimer(TIMER_HOST);
	for (int r = 0; r < ROW_SIZE; r++)
		for (int c = 0; c < COL_SIZE; c++)
			for (int k = 0; k < K_SIZE; k++)
				hostC[r][c] += A[r][k] * B[k][c];
	timer->offTimer(TIMER_HOST);

	// Copy input matrices : H -> D
	timer->onTimer(TIMER_HtoD);
	cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
	timer->offTimer(TIMER_HtoD);

	dim3 blockDim(ROW_SIZE, COL_SIZE);

	timer->onTimer(TIMER_KERNEL);
	matMul_kernel << <1, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	timer->offTimer(TIMER_KERNEL);

	timer->onTimer(TIMER_KERNEL_SH);
	matMul_kernel_shared << <1, blockDim >> > (dA, dB, dC);
	cudaDeviceSynchronize();
	timer->offTimer(TIMER_KERNEL_SH);

	// Get back result : D -> H
	timer->onTimer(TIMER_DtoH);
	cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
	timer->onTimer(TIMER_DtoH);

	// check the results
	bool isCorrect = true;

	float* pHostC = &hostC[0][0];
	float* pDeviceC = &deviceC[0][0];

	for (int i = 0; i < MAT_SIZE_C; i++) {
		if (pHostC[i] != pDeviceC[i]) {
			printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
			isCorrect = false;
			break;
		}
	}

	if (isCorrect)
		printf("Result is correct!\n");
	else
		printf("Result is not correct!!!!!!\n");

	timer->printTimer();
	if (timer != NULL)
		delete timer;
}

void genInputMatrices(void)
{
	for (int r = 0; r < ROW_SIZE; r++)
		for (int k = 0; k < K_SIZE; k++)
			A[r][k] = rand() % 100;

	for (int k = 0; k < K_SIZE; k++)
		for (int c = 0; c < COL_SIZE; c++)
			B[k][c] = rand() % 100;
}

void setTimer(void)
{
	timer = new DS_timer(NUM_TIMER);

	timer->initTimers();
	timer->setTimerName(TIMER_HOST, (char*)"CPU code");
	timer->setTimerName(TIMER_KERNEL, (char*)"Kernel launch");
	timer->setTimerName(TIMER_KERNEL_SH, (char*)"Kernel launch (shared ver.)");
	timer->setTimerName(TIMER_HtoD, (char*)"[Data transter] host->device");
	timer->setTimerName(TIMER_DtoH, (char*)"[Data transfer] device->host");
}
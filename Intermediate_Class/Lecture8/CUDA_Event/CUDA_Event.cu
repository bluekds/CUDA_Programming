#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE (64*1024*1024)
#define B_SIZE (1024)

#define NUM_STREAMS 1

__global__ void myKernel(int* _in, int* _out)
{
	int tID = blockDim.x * blockIdx.x + threadIdx.x;

	int temp = 0;
	for (int i = 0; i < 250; i++) {
		temp = (temp + _in[tID] * 5) % 10;
	}
	_out[tID] = temp;

}

void main(void)
{
	DS_timer timer(1);
	timer.setTimerName(0, "Total");

	int* in = NULL, * out = NULL, * out2 = NULL;

	cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE);
	memset(in, 0, sizeof(int) * ARRAY_SIZE);

	cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE);
	memset(out, 0, sizeof(int) * ARRAY_SIZE);

	cudaMallocHost(&out2, sizeof(int) * ARRAY_SIZE);
	memset(out2, 0, sizeof(int) * ARRAY_SIZE);

	int* dIn, * dOut;
	cudaMalloc(&dIn, sizeof(int) * ARRAY_SIZE);
	cudaMalloc(&dOut, sizeof(int) * ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; i++)
		in[i] = rand() % 10;

	// Multiple stream version
	cudaStream_t stream[NUM_STREAMS];
	cudaEvent_t start[NUM_STREAMS], end[NUM_STREAMS];

	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&stream[i]);
		cudaEventCreate(&start[i]); cudaEventCreate(&end[i]);
	}

	int chunkSize = ARRAY_SIZE / NUM_STREAMS;

	timer.onTimer(0);
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		int offset = chunkSize * i;
		cudaEventRecord(start[i], stream[i]);

		cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int) * chunkSize, cudaMemcpyHostToDevice, stream[i]);
		myKernel <<<chunkSize/B_SIZE, B_SIZE, 0, stream[i] >>> (dIn + offset, dOut + offset);
		cudaMemcpyAsync(out2 + offset, dOut + offset, sizeof(int) * chunkSize, cudaMemcpyDeviceToHost, stream[i]);

		cudaEventRecord(end[i], stream[i]);
	}

	cudaDeviceSynchronize();
	timer.offTimer(0);
	timer.printTimer();

	for (int i = 0; i < NUM_STREAMS; i++) {
		float time = 0;
		cudaEventElapsedTime(&time, start[i], end[i]);
		printf("Stream[%d] : %f ms\n", i, time);
	}

	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (out[i] != out2[i])
			printf("!");
	}

	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamDestroy(stream[i]);
		cudaEventDestroy(start[i]);
		cudaEventDestroy(end[i]);
	}

	cudaFree(dIn);
	cudaFree(dOut);

	cudaFreeHost(in);
	cudaFreeHost(out);
	cudaFreeHost(out2);
}
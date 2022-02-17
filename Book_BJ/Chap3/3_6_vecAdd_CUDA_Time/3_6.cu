#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The size of the vector
#define NUM_DATA 1024

// Simple vector sum kernel (Max vector size : 1024)
__global__ void vecAdd(int* _a, int* _b, int* _c) {
	int tID = threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void)
{
	// Set timer
	DS_timer timer(5);
	timer.setTimerName(0, (char*)"CUDA Total");
	timer.setTimerName(1, (char*)"Computation(Kernel)");
	timer.setTimerName(2, (char*)"Data Trans. : Host -> Device");
	timer.setTimerName(3, (char*)"Data Trans. : Device -> Host");
	timer.setTimerName(4, (char*)"VecAdd on Host");
	timer.initTimers();

	int* a, * b, * c, * hc;	// Vectors on the host
	int* da, * db, * dc;	// Vectors on the device

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

	// Memory allocation on the host-side
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	hc = new int[NUM_DATA]; memset(hc, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparision)
	timer.onTimer(4);
	for (int i = 0; i < NUM_DATA; i++)
		hc[i] = a[i] + b[i];
	timer.offTimer(4);

	// Memory allocation on the device-side
	cudaMalloc(&da, memSize); cudaMemset(da, 0, memSize);
	cudaMalloc(&db, memSize); cudaMemset(db, 0, memSize);
	cudaMalloc(&dc, memSize); cudaMemset(dc, 0, memSize);

	timer.onTimer(0);

	// Data copy : Host -> Device
	timer.onTimer(2);
	cudaMemcpy(da, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, memSize, cudaMemcpyHostToDevice);
	timer.offTimer(2);

	// Kernel call
	timer.onTimer(1);
	vecAdd <<<1, NUM_DATA >>> (da, db, dc);
	cudaDeviceSynchronize(); // synchronization function
	timer.offTimer(1);

	// Copy results : Device -> Host
	timer.onTimer(3);
	cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);
	timer.offTimer(3);

	timer.offTimer(0);

	// Release device memory
	cudaFree(da); cudaFree(db); cudaFree(dc);
	
	timer.printTimer();

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if (hc[i] != c[i]) {
			printf("[%d] The result is not matched! (%d, %d)\n"
				, i, hc[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	// Release host memory
	delete[] a; delete[] b; delete[] c;

	return 0;
}

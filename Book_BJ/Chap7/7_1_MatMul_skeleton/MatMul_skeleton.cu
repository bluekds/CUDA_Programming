#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DO_CPU
#define DATA_TYEP int

// Matrix size
#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

template<class T> void allocNinitMem(T** p, long long size, double* memUsage = NULL);
bool compareMatrix(DATA_TYEP* _A, DATA_TYEP* _B, int _size);

/******************************************************************
* Complete this kernels
******************************************************************/
__global__ void MatMul(DATA_TYEP* matA, DATA_TYEP* matB, DATA_TYEP* matC, int m, int n, int k)
{
	// Write your kernel here
}


int main(int argc, char* argv[])
{
	DS_timer timer(10);
	timer.setTimerName(0, (char*)"CPU code");
	timer.setTimerName(1, (char*)"Kernel");
	timer.setTimerName(2, (char*)"[Data transter] host->device");
	timer.setTimerName(3, (char*)"[Data transfer] device->host");
	timer.setTimerName(4, (char*)"GPU total");

	// set matrix size
	int m, n, k;
	m = SIZE_M;
	n = SIZE_N;
	k = SIZE_K;

	printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

	int sizeA = m * k;
	int sizeB = k * n;
	int sizeC = m * n;

	// Make matrix
	DATA_TYEP* A = NULL, * B = NULL;
	allocNinitMem<DATA_TYEP>(&A, sizeA);
	allocNinitMem<DATA_TYEP>(&B, sizeB);

	DATA_TYEP* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<DATA_TYEP>(&Ccpu, sizeC);
	allocNinitMem<DATA_TYEP>(&Cgpu, sizeC);

	// generate input matrices
	for (int i = 0; i < sizeA; i++) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	for (int i = 0; i < sizeB; i++) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

	// CPU algorithm
	timer.onTimer(0);
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int cIndex = row * n + col;
			Ccpu[cIndex] = 0;
			for (int i = 0; i < k; i++)
				Ccpu[cIndex] += (A[row * k + i] * B[i * n + col]);
		}
	}
	printf("CPU finished!\n");
	timer.offTimer(0);

	timer.onTimer(4);
	/******************************************************************
	* Write your codes for GPU algorithm from here
	******************************************************************/
	DATA_TYEP* dA, * dB, * dC;

	// 1. Allocate device memory for dA, dB, dC
	// Hint: cudaMalloc, cudaMemset

	timer.onTimer(2);

	// 2. Send(Copy) the input matrices to GPU (A -> dB, B -> dB)
	// Hint: cudaMemcpy

	timer.offTimer(2);

	// 3. Set the thread layout
	// 
	// dim3 gridDim(?, ?);
	// dim3 blockDim(?, ?);

	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	timer.onTimer(1);

	// 4. Kernel call
	//MatMul <<< gridDim, blockDim >>> (dA, dB, dC, m, n, k);

	cudaDeviceSynchronize(); // this is synchronization for mearusing the kernel processing time
	timer.offTimer(1);

	timer.onTimer(3);

	//5. Get(copy) the result from GPU to host memory (dC -> Cgpu)
	// Hint: cudaMemcpy

	timer.offTimer(3);

	// 6. Release device memory space (dA, dB, dC)
	// Hint: cudaFree


	/******************************************************************
	******************************************************************/
	timer.offTimer(4);

	compareMatrix(Ccpu, Cgpu, sizeC);
	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}


// Utility functions
bool compareMatrix(DATA_TYEP* _A, DATA_TYEP* _B, int _size)
{
	bool isMatched = true;
	for (int i = 0; i < _size; i++) {
		if (_A[i] != _B[i]) {
			printf("[%d] not matched! (%f, %f)\n", i, _A[i], _B[i]);
			getchar();
			isMatched = false;
		}
	}
	if (isMatched)
		printf("Results are matched!\n");
	else
		printf("Results are not matched!!!!!!!!!!!\n");

	return isMatched;
}

template<class T>
void allocNinitMem(T** p, long long size, double* memUsage) {
	*p = new T[size];
	memset(*p, 0, sizeof(T) * size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T) * size;
	}
}
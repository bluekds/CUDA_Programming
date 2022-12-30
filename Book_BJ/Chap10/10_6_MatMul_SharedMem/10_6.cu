#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_TYPE int

#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define INDEX2ROW(_index,_width)	(int)((_index)/(_width))
#define INDEX2COL(_index,_width)	((_index)%(_width))
#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))

#define BLOCK_SIZE 16

// macro function
#define IS_EQUAL(_a, _b) (abs(_b - _a) < 10e-6)

/******************************************************************
* Modify this kernel to use shared memory
******************************************************************/
__global__ void MatMul_SharedMem(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	DATA_TYPE val = 0;
	__shared__ DATA_TYPE subA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ DATA_TYPE subB[BLOCK_SIZE][BLOCK_SIZE];

	int localRow = threadIdx.x;
	int localCol = threadIdx.y;

	for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
		int offset = bID * BLOCK_SIZE;

		// load A and B
		if (row >= m || offset + localCol >= k)
			subA[localRow][localCol] = 0;
		else
			subA[localRow][localCol] = matA[row * k + (offset + localCol)];

		if (col >= n || offset + localRow >= k)
			subB[localRow][localCol] = 0;
		else
			subB[localRow][localCol] = matB[(offset + localRow) * n + col];

		__syncthreads();

		// compute
		for (int i = 0; i < BLOCK_SIZE; i++) {
			val += subA[localRow][i] * subB[i][localCol];
		}
		__syncthreads();
	}

	if (row >= m || col >= n)
		return;

	matC[row * n + col] = val;
}
/******************************************************************
******************************************************************/

template<class T> void allocNinitMem(T** p, long long size, DATA_TYPE* memUsage = NULL);
void runMatMul_Basic(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k);
bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size);

DS_timer timer(10);
void setTimer();

int main(int argc, char* argv[])
{
	setTimer();

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
	DATA_TYPE* A = NULL, * B = NULL;
	allocNinitMem<DATA_TYPE>(&A, sizeA);
	allocNinitMem<DATA_TYPE>(&B, sizeB);

	DATA_TYPE* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<DATA_TYPE>(&Ccpu, sizeC);
	allocNinitMem<DATA_TYPE>(&Cgpu, sizeC);

	// generate input matrices
	for (int i = 0; i < sizeA; i++) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	for (int i = 0; i < sizeB; i++) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

	// CPU version (OpenMP)
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

	// GPU setup
	DATA_TYPE* dA, * dB, * dC;

	cudaMalloc(&dA, sizeA * sizeof(DATA_TYPE));
	cudaMemset(dA, 0, sizeA * sizeof(DATA_TYPE));

	cudaMalloc(&dB, sizeB * sizeof(DATA_TYPE));
	cudaMemset(dB, 0, sizeB * sizeof(DATA_TYPE));

	cudaMalloc(&dC, sizeC * sizeof(DATA_TYPE));
	cudaMemset(dC, 0, sizeC * sizeof(DATA_TYPE));

	timer.onTimer(1);

	timer.onTimer(4);
	cudaMemcpy(dA, A, sizeA * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
	timer.offTimer(4);

	/******************************************************************
	* Write your codes for GPU algorithm from here
	******************************************************************/
	// Sharead memroy version

	// 1. set the thread layout
	// Change the layout if you need
	dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	// 2. kernel call
	timer.onTimer(3);
	MatMul_SharedMem <<<gridDim, blockDim >>> (dA, dB, dC, m, n, k);
	cudaDeviceSynchronize();
	timer.offTimer(3);

	/******************************************************************
	******************************************************************/

	timer.onTimer(5);
	cudaMemcpy(Cgpu, dC, sizeC * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
	timer.offTimer(5);

	timer.offTimer(1);

	// Basci version
	runMatMul_Basic(dA, dB, dC, m, n, k);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	printf("[Kernel (shared memroy)] ");
	compareMatrix(Ccpu, Cgpu, sizeC);

	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}

bool compareMatrix(DATA_TYPE* _A, DATA_TYPE* _B, int _size)
{
	bool isMatched = true;
	for (int i = 0; i < _size; i++) {
		if (!IS_EQUAL(_A[i], _B[i])) {
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

__global__ void MatMul(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row >= m || col >= n)
		return;

	DATA_TYPE val = 0; // hope to use register
	for (int i = 0; i < k; i++)
		val += matA[ID2INDEX(row, i, k)] * matB[ID2INDEX(i, col, n)];

	matC[ID2INDEX(row, col, n)] = val;
}

void runMatMul_Basic(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
	dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	timer.onTimer(7);
	MatMul <<< gridDim, blockDim >>> (matA, matB, matC, m, n, k);
	cudaDeviceSynchronize();
	timer.offTimer(7);

	cudaMemset(matC, 0, m * n * sizeof(DATA_TYPE));
}

template<class T>
void allocNinitMem(T** p, long long size, DATA_TYPE* memUsage) {
	*p = new T[size];
	memset(*p, 0, sizeof(T) * size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T) * size;
	}
}

void setTimer()
{
	timer.setTimerName(0, (char*)"CPU algorithm");
	timer.setTimerName(1, (char*)"GPU/CUDA algorithm");
	timer.setTimerName(3, (char*)" - Kernel (Shared memory)");
	timer.setTimerName(4, (char*)" - [Data transfer] host->device");
	timer.setTimerName(5, (char*)" - [Data transfer] device->host");
	timer.setTimerName(7, (char*)"Kernel (Basic)");
}
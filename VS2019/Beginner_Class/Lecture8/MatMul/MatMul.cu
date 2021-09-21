/**
This is an exmple solution code for Lab. 5 and Lab. 6-1  Matrix Multiplication <br>
@author : Duksu Kim
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <DS_timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DO_CPU
#define DO_VER2
#define DO_VER3

#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define INDEX2ROW(_index,_width)	(int)((_index)/(_width))
#define INDEX2COL(_index,_width)	((_index)%(_width))
#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))

#define BLOCK_SIZE 16

// Macro function
#define LOOP_I(_size) for(int i = 0 ; i < _size; i++)
#define KERNEL_MUL(_a,_b) __fmul_rn(_a,_b)
//#define KERNEL_MUL(_a,_b) (_a*_b)

// kernel declarations
__global__ void MatMul(float* matA, float* matB, float* matC, int m, int n, int k);
__global__ void MatMul_shared(float* matA, float* matB, float* matC, int m, int n, int k);
__global__ void MatMul_shared_NobankConflict(float* matA, float* matB, float* matC, int m, int n, int k);

template<class T> void allocNinitMem(T** p, long long size, double* memUsage = NULL);
bool compareMatrix(float* _A, float* _B, int _size);

int main(int argc, char* argv[])
{
	DS_timer timer(10);
	timer.setTimerName(0, "CPU code");
	timer.setTimerName(1, "Kernel - basic");
	timer.setTimerName(2, "Kernel - shared memory");
	timer.setTimerName(3, "Kernel - no bank conflict");
	timer.setTimerName(4, "[Data transter] host->device");
	timer.setTimerName(5, "[Data transfer] device->host");

	// set matrix size
	int m, n, k;

	if (argc < 3) { m = SIZE_M;	n = SIZE_N;	k = SIZE_K; }
	else { m = atoi(argv[1]);	n = atoi(argv[2]);	k = atoi(argv[3]); }

	printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

	int sizeA = m * k;
	int sizeB = k * n;
	int sizeC = m * n;

	// Make matrix
	float* A = NULL, * B = NULL;
	allocNinitMem<float>(&A, sizeA);
	allocNinitMem<float>(&B, sizeB);

	float* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<float>(&Ccpu, sizeC);
	allocNinitMem<float>(&Cgpu, sizeC);

	// generate input matrices
	LOOP_I(sizeA) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	LOOP_I(sizeB) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

#ifdef DO_CPU // CPU version (OpenMP)
	timer.onTimer(0);
#pragma omp parallel for num_threads(4)
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int cIndex = ID2INDEX(row, col, n);
			Ccpu[cIndex] = 0;
			LOOP_I(k)
				Ccpu[cIndex] += (A[ID2INDEX(row, i, k)] * B[ID2INDEX(i, col, n)]);
		}
	}
	printf("CPU finished!\n");
	timer.offTimer(0);
#endif

	// GPU setup
	float* dA, * dB, * dC;

	checkCudaErrors(cudaMalloc(&dA, sizeA * sizeof(float)));
	checkCudaErrors(cudaMemset(dA, 0, sizeA * sizeof(float)));

	checkCudaErrors(cudaMalloc(&dB, sizeB * sizeof(float)));
	checkCudaErrors(cudaMemset(dB, 0, sizeB * sizeof(float)));

	checkCudaErrors(cudaMalloc(&dC, sizeC * sizeof(float)));
	checkCudaErrors(cudaMemset(dC, 0, sizeC * sizeof(float)));

	timer.onTimer(4);
	checkCudaErrors(cudaMemcpy(dA, A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dB, B, sizeB * sizeof(float), cudaMemcpyHostToDevice));
	timer.offTimer(4);

	dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	// GPU version 1 (basic)
	timer.onTimer(1);
	MatMul << <gridDim, blockDim >> > (dA, dB, dC, m, n, k);
	checkCudaErrors(cudaDeviceSynchronize());
	timer.offTimer(1);

	timer.onTimer(5);
	checkCudaErrors(cudaMemcpy(Cgpu, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
	timer.offTimer(5);

#ifdef DO_CPU
	printf("[Kernel basic] ");
	compareMatrix(Ccpu, Cgpu, sizeC);
#endif

#ifdef DO_VER2 // GPU version 2 (shared memory)
	timer.onTimer(2);
	MatMul_shared << <gridDim, blockDim >> > (dA, dB, dC, m, n, k);
	checkCudaErrors(cudaDeviceSynchronize());
	timer.offTimer(2);

	checkCudaErrors(cudaMemcpy(Cgpu, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef DO_CPU
	printf("[Kernel shared memory] ");
	compareMatrix(Ccpu, Cgpu, sizeC);
#endif
#endif

#ifdef DO_VER3 // GPU version 3 (shared memory + no bank conflict)
	timer.onTimer(3);
	MatMul_shared_NobankConflict << <gridDim, blockDim >> > (dA, dB, dC, m, n, k);
	checkCudaErrors(cudaDeviceSynchronize());
	timer.offTimer(3);

	checkCudaErrors(cudaMemcpy(Cgpu, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef DO_CPU
	printf("[Kernel no bank conflict] ");
	compareMatrix(Ccpu, Cgpu, sizeC);
#endif
#endif

	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	checkCudaErrors(cudaFree(dA));
	checkCudaErrors(cudaFree(dB));
	checkCudaErrors(cudaFree(dC));

	return 0;
}

bool compareMatrix(float* _A, float* _B, int _size)
{
	bool isMatched = true;
	LOOP_I(_size) {
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

__global__ void MatMul(float* matA, float* matB, float* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row >= m || col >= n)
		return;

	float val = 0; // hope to use register
	LOOP_I(k)
		val += KERNEL_MUL(matA[ID2INDEX(row, i, k)], matB[ID2INDEX(i, col, n)]);

	matC[ID2INDEX(row, col, n)] = val;
}

__global__ void MatMul_shared(float* matA, float* matB, float* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	float val = 0;
	__shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];

	int localRow = threadIdx.x;
	int localCol = threadIdx.y;

	for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
		int offset = bID * BLOCK_SIZE;

		// load A and B
		if (row >= m || offset + localCol >= k)
			subA[localRow][localCol] = 0;
		else
			subA[localRow][localCol] = matA[ID2INDEX(row, offset + localCol, k)];

		if (col >= n || offset + localRow >= k)
			subB[localRow][localCol] = 0;
		else
			subB[localRow][localCol] = matB[ID2INDEX(offset + localRow, col, n)];

		__syncthreads();

		// compute
		LOOP_I(BLOCK_SIZE) {
			val += KERNEL_MUL(subA[localRow][i], subB[i][localCol]);
		}
		__syncthreads();
	}

	if (row >= m || col >= n)
		return;

	matC[ID2INDEX(row, col, n)] = val;
}

__global__ void MatMul_shared_NobankConflict(float* matA, float* matB, float* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	float val = 0;
	__shared__ float subA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float subB[BLOCK_SIZE][BLOCK_SIZE];

	int localRow = threadIdx.x;
	int localCol = threadIdx.y;

	for (int bID = 0; bID < ceil((float)k / BLOCK_SIZE); bID++) {
		int offset = bID * BLOCK_SIZE;

		// load A and B
		if (row >= m || offset + localCol >= k)
			subA[localCol][localRow] = 0;
		else
			subA[localCol][localRow] = matA[ID2INDEX(row, offset + localCol, k)];

		if (col >= n || offset + localRow >= k)
			subB[localRow][localCol] = 0;
		else
			subB[localRow][localCol] = matB[ID2INDEX(offset + localRow, col, n)];

		__syncthreads();

		// compute
		LOOP_I(BLOCK_SIZE) {
			val += KERNEL_MUL(subA[i][localRow], subB[i][localCol]);
		}
		__syncthreads();
	}

	if (row >= m || col >= n)
		return;

	matC[ID2INDEX(row, col, n)] = val;
}

template<class T>
void allocNinitMem(T** p, long long size, double* memUsage) {
	*p = new T[size];
	memset(*p, 0, sizeof(T) * size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T) * size;
	}
}
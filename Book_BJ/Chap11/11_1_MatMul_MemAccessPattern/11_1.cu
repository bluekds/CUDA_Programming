#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define BLOCK_SIZE 32

// kernels
__global__ void MatMul_xRow(int* matA, int* matB, int* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row >= m || col >= n)
		return;

	int val = 0;
	for (int i = 0; i < k; i++)
		val += matA[row * k + i] * matB[i * n + col];

	matC[row * n + col, n] = val;
}

__global__ void MatMul_yRow(int* matA, int* matB, int* matC, int m, int n, int k)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row >= m || col >= n)
		return;

	int val = 0;
	for (int i = 0; i < k; i++)
		val += matA[row * k + i] * matB[i * n + col];

	matC[row * n + col, n] = val;
}

template<class T> void allocNinitMem(T** p, long long size, double* memUsage = NULL);
bool compareMatrix(int* _A, int* _B, int _size);

int main(int argc, char* argv[])
{
	DS_timer timer(10);
	timer.setTimerName(1, (char*)" - X-dim = Row Kernel");
	timer.setTimerName(2, (char*)" - Y-dim = Row Kernel");

	// set matrix size
	int m, n, k;

	if (argc < 3) { m = SIZE_M;	n = SIZE_N;	k = SIZE_K; }
	else { m = atoi(argv[1]);	n = atoi(argv[2]);	k = atoi(argv[3]); }

	printf("Size : A = (%d by %d), B = (%d by %d), C = (%d by %d)\n", m, k, k, n, m, n);

	int sizeA = m * k;
	int sizeB = k * n;
	int sizeC = m * n;

	// Make matrix
	int* A = NULL, * B = NULL;
	allocNinitMem<int>(&A, sizeA);
	allocNinitMem<int>(&B, sizeB);

	int* Ccpu = NULL, * Cgpu = NULL;
	allocNinitMem<int>(&Ccpu, sizeC);
	allocNinitMem<int>(&Cgpu, sizeC);

	// generate input matrices
	for (int i = 0; i < sizeA; i++) A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
	for (int i = 0; i < sizeB; i++) B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

	// GPU setup
	int* dA, * dB, * dC;

	cudaMalloc(&dA, sizeA * sizeof(int));
	cudaMemset(dA, 0, sizeA * sizeof(int));

	cudaMalloc(&dB, sizeB * sizeof(int));
	cudaMemset(dB, 0, sizeB * sizeof(int));

	cudaMalloc(&dC, sizeC * sizeof(int));
	cudaMemset(dC, 0, sizeC * sizeof(int));

	cudaMemcpy(dA, A, sizeA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB * sizeof(int), cudaMemcpyHostToDevice);

	// Row = X-dim version
	timer.onTimer(1);
	dim3 gridDim_xRow(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
	dim3 blockDim_xRow(BLOCK_SIZE, BLOCK_SIZE);
	MatMul_xRow << <gridDim_xRow, blockDim_xRow >> > (dA, dB, dC, m, n, k);
	cudaDeviceSynchronize();
	timer.offTimer(1);

	// Row = Y-dim version
	timer.onTimer(2);
	dim3 gridDim_yRow(ceil((float)n / BLOCK_SIZE), ceil((float)m / BLOCK_SIZE));
	dim3 blockDim_yRow(BLOCK_SIZE, BLOCK_SIZE);
	MatMul_yRow << <gridDim_yRow, blockDim_yRow >> > (dA, dB, dC, m, n, k);
	cudaDeviceSynchronize();
	timer.offTimer(2);

	cudaMemcpy(Cgpu, dC, sizeC * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}

template<class T>
void allocNinitMem(T** p, long long size, double* memUsage) {
	*p = new T[size];
	memset(*p, 0, sizeof(T) * size);

	if (memUsage != NULL) {
		*memUsage += sizeof(T) * size;
	}
}
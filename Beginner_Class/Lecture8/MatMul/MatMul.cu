#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DO_CPU
#define DATA_TYEP int

#define SIZE_M (512*2)
#define SIZE_N (512*4)
#define SIZE_K (512*2)

#define INDEX2ROW(_index,_width)	(int)((_index)/(_width))
#define INDEX2COL(_index,_width)	((_index)%(_width))
#define ID2INDEX(_row,_col, _width) (((_row)*(_width))+(_col))

#define BLOCK_SIZE 16

// Macro function
//#define KERNEL_MUL(_a,_b) __fmul_rn(_a,_b)
#define KERNEL_MUL(_a,_b) (_a*_b)

// kernel declarations
__global__ void MatMul(DATA_TYEP* matA, DATA_TYEP* matB, DATA_TYEP* matC, int m, int n, int k);

template<class T> void allocNinitMem(T** p, long long size, double* memUsage = NULL);
bool compareMatrix(DATA_TYEP* _A, DATA_TYEP* _B, int _size);

int main(int argc, char* argv[])
{
	DS_timer timer(10);
	timer.setTimerName(0, (char*)"CPU algorithm");
	timer.setTimerName(1, (char*)"GPU/CUDA algorithm");
	timer.setTimerName(2, (char*)" - Kernel");
	timer.setTimerName(4, (char*)" - [Data transter] host->device");
	timer.setTimerName(5, (char*)" - [Data transfer] device->host");

	// set matrix size
	int m, n, k;

	if (argc < 3) { m = SIZE_M;	n = SIZE_N;	k = SIZE_K; }
	else { m = atoi(argv[1]);	n = atoi(argv[2]);	k = atoi(argv[3]); }

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

#ifdef DO_CPU // CPU version (OpenMP)
	timer.onTimer(0);
#pragma omp parallel for num_threads(4)
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			int cIndex = ID2INDEX(row, col, n);
			Ccpu[cIndex] = 0;
			for (int i = 0; i < k ; i++)
				Ccpu[cIndex] += (A[ID2INDEX(row, i, k)] * B[ID2INDEX(i, col, n)]);
		}
	}
	printf("CPU finished!\n");
	timer.offTimer(0);
#endif

	// GPU setup
	DATA_TYEP* dA, * dB, * dC;

	cudaMalloc(&dA, sizeA * sizeof(DATA_TYEP));
	cudaMemset(dA, 0, sizeA * sizeof(DATA_TYEP));

	cudaMalloc(&dB, sizeB * sizeof(DATA_TYEP));
	cudaMemset(dB, 0, sizeB * sizeof(DATA_TYEP));

	cudaMalloc(&dC, sizeC * sizeof(DATA_TYEP));
	cudaMemset(dC, 0, sizeC * sizeof(DATA_TYEP));

	timer.onTimer(1);

	timer.onTimer(4);
	cudaMemcpy(dA, A, sizeA * sizeof(DATA_TYEP), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeB * sizeof(DATA_TYEP), cudaMemcpyHostToDevice);
	timer.offTimer(4);

	dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

	printf("Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

	// GPU version
	timer.onTimer(2);
	MatMul <<< gridDim, blockDim >>> (dA, dB, dC, m, n, k);
	cudaDeviceSynchronize();
	timer.offTimer(2);

	timer.onTimer(5);
	cudaMemcpy(Cgpu, dC, sizeC * sizeof(DATA_TYEP), cudaMemcpyDeviceToHost);
	timer.offTimer(5);

	timer.offTimer(1);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

#ifdef DO_CPU
	printf("[Kernel basic] ");
	compareMatrix(Ccpu, Cgpu, sizeC);
#endif

	timer.printTimer(1);

	delete A;
	delete B;
	delete Ccpu;
	delete Cgpu;

	return 0;
}

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

__global__ void MatMul(DATA_TYEP* matA, DATA_TYEP* matB, DATA_TYEP* matC, int m, int n, int k)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row >= m || col >= n)
		return;

	DATA_TYEP val = 0; // hope to use register
	for (int i = 0; i < k ; i++)
		val += KERNEL_MUL(matA[ID2INDEX(row, i, k)], matB[ID2INDEX(i, col, n)]);

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
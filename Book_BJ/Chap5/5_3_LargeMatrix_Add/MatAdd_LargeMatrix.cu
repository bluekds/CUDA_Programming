#include "MatAdd_LargeMatrix.h"

__global__ void MatAdd_G2D_B2D
(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE)
{
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int index = row * COL_SIZE + col;

	if (col < COL_SIZE && row < ROW_SIZE)
		MatC[index] = MatA[index] + MatB[index];
}

__global__ void MatAdd_G1D_B1D
(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE)
{
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x; // col
	if (col < COL_SIZE) {
		for (int row = 0; row < ROW_SIZE; row++) {
			int index = row * COL_SIZE + col;
			MatC[index] = MatA[index] + MatB[index];
		}
	}
}

__global__ void MatAdd_G2D_B1D
(float* MatA, float* MatB, float* MatC, int ROW_SIZE, int COL_SIZE)
{
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x; // col
	unsigned int row = blockIdx.y;							 // row
	unsigned int index = row * COL_SIZE + col;

	if (col < COL_SIZE && row < ROW_SIZE)
		MatC[index] = MatA[index] + MatB[index];
}

bool kernelCall(float* _MatA, float* _MatB, float* _MatC, int _ROW_SIZE, int _COL_SIZE
	, int _layout, dim3 _gridDim, dim3 _blockDim)
{
	switch (_layout)
	{
	case ThreadLayout::G1D_B1D:
		MatAdd_G1D_B1D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _ROW_SIZE, _COL_SIZE);
		break;
	case ThreadLayout::G2D_B1D:
		MatAdd_G2D_B1D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _ROW_SIZE, _COL_SIZE);
		break;
	case ThreadLayout::G2D_B2D:
		MatAdd_G2D_B2D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _ROW_SIZE, _COL_SIZE);
		break;
	default:
		printf("Not supported layout\n");
		return false;
	}
	return true;
}
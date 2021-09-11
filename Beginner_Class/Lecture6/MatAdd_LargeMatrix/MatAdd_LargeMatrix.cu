/**
This is an exmple solution code for Lab 4-2. Vector sum for large vectors <br>
@author : Duksu Kim
*/

#include "MatAdd_LargeMatrix.h"

__global__ void MatAdd_G2D_B2D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // col
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; // row
	unsigned int idx = iy * nCol + ix;

	if (ix < nCol && iy < nRow)
		MatC[idx] = MatA[idx] + MatB[idx];
}

__global__ void MatAdd_G1D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // col
	if (ix < nCol) {
		for (int iy = 0; iy < nRow; iy++) {
			int idx = iy * nCol + ix;
			MatC[idx] = MatA[idx] + MatB[idx];
		}
	}
}

__global__ void MatAdd_G2D_B1D
(float* MatA, float* MatB, float* MatC, int nRow, int nCol)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // col
	unsigned int iy = blockIdx.y;							 // row
	unsigned int idx = iy * nCol + ix;

	if (ix < nCol && iy < nRow)
		MatC[idx] = MatA[idx] + MatB[idx];
}

bool kernelCall(float* _MatA, float* _MatB, float* _MatC, int _nRow, int _nCol
	, int _layout, dim3 _gridDim, dim3 _blockDim)
{
	switch (_layout)
	{
	case ThreadLayout::G1D_B1D:
		MatAdd_G1D_B1D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _nRow, _nCol);
		break;
	case ThreadLayout::G2D_B1D:
		MatAdd_G2D_B1D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _nRow, _nCol);
		break;
	case ThreadLayout::G2D_B2D:
		MatAdd_G2D_B2D <<<_gridDim, _blockDim >>> (_MatA, _MatB, _MatC, _nRow, _nCol);
		break;
	default:
		printf("Not supported layout\n");
		return false;
	}
	return true;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum ThreadLayout
{
	G1D_B1D, G2D_B1D, G2D_B2D, NUM_LAYOUTS
};

/**
Interface function for kernel call
*/
bool kernelCall
(float* _MatA, float* _MatB, float* _MatC, int _nRow, int _nCol
	, int _layout, dim3 _griDim, dim3 _blockDim);
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
	int* a, * b, * c, * h_c;	// Vectors on the host
	int* d_a, * d_b, * d_c;		// Vectors on the device

	int memSize = sizeof(int) * NUM_DATA;
	printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

	// Memory allocation on the host-side
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	h_c = new int[NUM_DATA]; memset(h_c, 0, memSize);

	// Data generation
	for (int i = 0; i < NUM_DATA; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	// Vector sum on host (for performance comparision)
	for (int i = 0; i < NUM_DATA; i++)
		h_c[i] = a[i] + b[i];

	//****************************************//
	//******* Write your code - start ********//

	// 1. Memory allocation on the device-side (d_a, d_b, d_c)

	// 2. Data copy : Host (a, b) -> Device (d_a, d_b)

	// 3. Kernel call
	// vecAdd << <1, NUM_DATA >> > (d_a, d_b, d_c);

	// 4. Copy results : Device (d_c) -> Host (c)

	// 5. Release device memory (d_a, d_b, d_c)

	//******** Write your code - end *********//
	//****************************************//

	// Check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++) {
		if (h_c[i] != c[i]) {
			printf("[%d] The resutl is not matched! (%d, %d)\n"
				, i, h_c[i], c[i]);
			result = false;
		}
	}

	if (result)
		printf("GPU works well!\n");

	// Release host memory
	delete[] a; delete[] b; delete[] c;

	return 0;
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void main(void) {
	int ngpus;
	cudaGetDeviceCount(&ngpus);

	for (int i = 0; i < ngpus; i++) {
		cudaDeviceProp devProp;

		cudaGetDeviceProperties(&devProp, i);
		printf("Device[%d](%s) compute capability : %d.%d.\n"
			, i, devProp.name, devProp.major, devProp.minor);
	}
}
#include "kernelCall.h"

void main() {
	kernelCall();
	printf("Host code running on CPU\n");
	cudaDeviceSynchronize();
}

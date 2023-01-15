#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_SIZE 64

__global__ void syncWarp_test()
{
    int tID = threadIdx.x;
    int warpID = (int)(tID / 32);
    __shared__ int masterID[BLOCK_SIZE/32];

    if (threadIdx.x % 32 == 0) {
        masterID[warpID] = tID;
    }
    __syncwarp(); // intra-warp synchronization (barrier)

    printf("[t%d] The master of our warp is t%d\n", tID, masterID[warpID]);
}

int main()
{
    syncWarp_test <<<1, BLOCK_SIZE >>>();
}
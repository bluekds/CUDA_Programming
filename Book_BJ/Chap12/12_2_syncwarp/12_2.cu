#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_SIZE 64

__global__ void syncWarp_test()
{
    int tID = threadIdx.x;
    int warpID = (int)(tID / 32);
    __shared__ int a[BLOCK_SIZE/32];

    if (threadIdx.x % 32 == 0) {
        a[warpID] = tID;
    }
    __syncwarp(); // intra-warp synchronization (barrier)

    printf("[%d] warp = %d\n", tID, a[warpID]);
}

int main()
{
    syncWarp_test <<<1, BLOCK_SIZE >>>();
}
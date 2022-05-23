// Block ID
#define	BID_X	blockIdx.x
#define	BID_Y	blockIdx.y
#define	BID_Z	blockIdx.z

// Thread ID
#define	TID_X	threadIdx.x
#define	TID_Y	threadIdx.y
#define	TID_Z	threadIdx.z

// Dimension of a grid
#define Gdim_X	gridDim.x
#define Gdim_Y	gridDim.y
#define Gdim_Z	gridDim.z

// Dimension of a block
#define Bdim_X	blockDim.x
#define Bdim_Y	blockDim.y
#define Bdim_Z	blockDim.z

#define TID_IN_BLOCK	(TID_Z*(Bdim_Y*Bdim_X) + TID_Y*Bdim_X + TID_X)
#define NUM_THREAD_IN_BLOCK	(Bdim_X*Bdim_Y*Bdim_Z)

#define GRID_1D_TID (BID_X * NUM_THREAD_IN_BLOCK) + TID_IN_BLOCK
#define GRID_2D_TID (BID_Y * (Gdim_X * NUM_THREAD_IN_BLOCK) + GRID_1D_TID)
#define GLOBAL_TID (BID_Z * (Gdim_Y * Gdim_X * NUM_THREAD_IN_BLOCK) + GRID_2D_TID)
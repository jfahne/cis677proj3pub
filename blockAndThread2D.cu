//  blockAndThread2D.c
//  2-D version of blockAndThread
//  fills two "flattened" 2D arrays of integers with block id and thread id


#include <iostream>
#include <cstdlib>
using namespace std;

// using 8x8 array
#define ARRAY_DIM 8 
#define BLOCK_SIZE 4


//  kernel function
__global__ void cu_fillArray(int *block_dX, int *block_dY, int *thread_dX, int *thread_dY, int dimArray)
{
	int x;
	int y;
	int pos;

	// compute x,y index and global position of thread
	x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	pos = x * dimArray + y;
	
	// appropriately fill arrays
	block_dX[pos] = blockIdx.x;
	block_dY[pos] = blockIdx.y;
	thread_dX[pos] = threadIdx.x;
	thread_dY[pos] = threadIdx.y;
}


int main (int argc, char *argv[])
{
    // declare arrays and initialize to 0
    int blockIdX[ARRAY_DIM*ARRAY_DIM];
    int blockIdY[ARRAY_DIM*ARRAY_DIM];
    int threadIdX[ARRAY_DIM*ARRAY_DIM];
    int threadIdY[ARRAY_DIM*ARRAY_DIM];

	for (int i=0; i < ARRAY_DIM; i++) {
        for (int j=0; j < ARRAY_DIM; j++) {
            blockIdX[i * ARRAY_DIM + j] = 0;
            blockIdY[i * ARRAY_DIM + j] = 0;
            threadIdX[i * ARRAY_DIM + j] = 0;
            threadIdY[i * ARRAY_DIM + j] = 0;
        }
    }

	int *block_dx, *block_dy, *thread_dx, *thread_dy;
	int nBlocks;
	cudaError_t result;

	// allocate memory on device
	result = cudaMalloc((void**) &block_dx, sizeof(int) * ARRAY_DIM*ARRAY_DIM);
	result = cudaMalloc((void**) &block_dy, sizeof(int) * ARRAY_DIM*ARRAY_DIM);
	result = cudaMalloc((void**) &thread_dx, sizeof(int) * ARRAY_DIM*ARRAY_DIM);
	result = cudaMalloc((void**) &thread_dy, sizeof(int) * ARRAY_DIM*ARRAY_DIM);
	if (result != cudaSuccess) {
		printf("cudaMalloc failed\n");
		exit(1);
	}

	// compute dimensions; assumes array size is a multiple of the block size
	dim3 dimblock (BLOCK_SIZE, BLOCK_SIZE);
	nBlocks = ARRAY_DIM/BLOCK_SIZE;
	dim3 dimgrid (nBlocks, nBlocks);

	// call the kernel
	cu_fillArray <<<dimgrid,dimblock>>> (block_dx, block_dy, thread_dx, thread_dy, ARRAY_DIM);

	// transfer results back to host, cleanup M
	result = cudaMemcpy(blockIdX, block_dx, sizeof(int) * ARRAY_DIM*ARRAY_DIM, cudaMemcpyDeviceToHost);
	result = cudaMemcpy(blockIdY, block_dy, sizeof(int) * ARRAY_DIM*ARRAY_DIM, cudaMemcpyDeviceToHost);
	result = cudaMemcpy(threadIdX, thread_dx, sizeof(int) * ARRAY_DIM*ARRAY_DIM, cudaMemcpyDeviceToHost);
	result = cudaMemcpy(threadIdY, thread_dy, sizeof(int) * ARRAY_DIM*ARRAY_DIM, cudaMemcpyDeviceToHost);
	cudaFree(block_dx);
	cudaFree(block_dy);
	cudaFree(thread_dx);
	cudaFree(thread_dy);
	if (result != cudaSuccess) {
		printf("cudaMemcpy - GPU to host - failed\n");
		exit(1);
	}

    // print arrays
    cout << "Final state of the blockIdX array:" << endl;
	for (int i = 0; i < ARRAY_DIM; i++) {
        for (int j = 0; j < ARRAY_DIM; j++) {
            cout << blockIdX[i * ARRAY_DIM + j] << " ";
        }        
        cout << endl;
    }
    cout << "Final state of the blockIdY array:" << endl;
	for (int i = 0; i < ARRAY_DIM; i++) {
        for (int j = 0; j < ARRAY_DIM; j++) {
            cout << blockIdY[i * ARRAY_DIM + j] << " ";
        }        
        cout << endl;
    }
    cout << "Final state of the threadIdX array:" << endl;
	for (int i = 0; i < ARRAY_DIM; i++) {
        for (int j = 0; j < ARRAY_DIM; j++) {
            cout << threadIdX[i * ARRAY_DIM + j] << " ";
        }        
        cout << endl;
    }
    cout << "Final state of the threadIdY array:" << endl;
	for (int i = 0; i < ARRAY_DIM; i++) {
        for (int j = 0; j < ARRAY_DIM; j++) {
            cout << threadIdY[i * ARRAY_DIM + j] << " ";
        }        
        cout << endl;
    }
 
    return 0;
}

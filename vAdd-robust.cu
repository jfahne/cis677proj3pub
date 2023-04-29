// vAdd-soln.cu
// GPU vector add (fixed block and memory issues)

#include <iostream>
using namespace std;

#define ARRAY_SIZE 63
#define BLOCK_SIZE 32


__global__ void cu_vAdd (int *a_d, int *b_d, int *c_d, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	// (x<n) ensures that only existing array elements get processed
	if (x < n)
		c_d[x] = a_d[x] + b_d[x];
}


int main (int argc, char *argv[])
{
	int *a = new int[ARRAY_SIZE];
	int *b = new int[ARRAY_SIZE];
	int *c = new int[ARRAY_SIZE];
	for (int i=0; i < ARRAY_SIZE; i++) {
		a[i] = i+1;
		b[i] = i+1;
		c[i] = 0;
	}

	int *a_d, *b_d, *c_d;
	cudaMalloc ((void**) &a_d, sizeof(int) * ARRAY_SIZE);
	cudaMalloc ((void**) &b_d, sizeof(int) * ARRAY_SIZE);
	cudaMalloc ((void**) &c_d, sizeof(int) * ARRAY_SIZE);
	cudaMemcpy (a_d, a, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy (b_d, b, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy (c_d, c, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	// ensures ample blocks of threads get created
	cu_vAdd <<< ceil ((float) ARRAY_SIZE/BLOCK_SIZE), BLOCK_SIZE >>> (a_d, b_d, c_d, ARRAY_SIZE);

	cudaMemcpy (c, c_d, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
	cudaFree (a_d);
	cudaFree (b_d);
	cudaFree (c_d);

	cout << "Array c:" << endl;
	for (int i=0; i < ARRAY_SIZE; i++)
		cout << c[i] << " ";
	cout << endl;
  
	delete[]a;
	delete[]b;
	delete[]c;
  
	return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <assert.h>

#define ROD_LENGTH 1.0f
#define TIME_STEP 1e-8
#define T_FINAL 0.1

__global__ void heat_diffusion(float *u, float *u_new, int num_slices)
{
    extern __shared__ float shared_mem[];
    float *u_shared = shared_mem;
    float *u_new_shared = &shared_mem[num_slices];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left_idx = (idx == 0) ? idx : idx - 1;
    int right_idx = (idx == (num_slices - 1)) ? idx : idx + 1;

    if (idx < num_slices) {
        atomicExch(&u_shared[idx],u[idx]);
    	__threadfence();
	float le_a = (100+u_shared[right_idx]);
    	float le_b = ((idx == 0)?1.0f:0)*le_a;
	float m_a = (u_shared[left_idx] + u_shared[right_idx]);
	float m_b = (((idx>=1) and (idx<=(num_slices-2))) ? 1.0f : 0) * m_a;
	float re_a = (u_shared[left_idx]+23.0f); 
	float re_b = ((idx==(num_slices-1))?1.0f:0)*(u_shared[left_idx]+23.0f); 
	float val = (re_b + m_b + le_b)/2;
	val = (val >= u_shared[idx]) ? val : u_shared[idx];	
	val = (val >= 100.0f) ? 100.0f : val;
	u_new_shared[idx] = val;
    	atomicExch(&u_new[idx], u_new_shared[idx]);
    } else {
	    ;
    }
}

int main()
{
    const int num_slices = 2500;
    const int num_steps = T_FINAL / TIME_STEP;
    const float dx = ROD_LENGTH / num_slices;
    const int plot_idx = (int)(0.3/dx);

    float *u = (float *)malloc(num_slices * sizeof(float));
    float *u_new = (float *)malloc(num_slices * sizeof(float));
    float *d_u;
    float *d_u_new;

    cudaError_t err = cudaMalloc((void **)&d_u, num_slices * sizeof(float));
    printf("%s\n",cudaGetErrorString(err));
    cudaMalloc((void **) &d_u_new, num_slices * sizeof(float));

    // Initialize temperature at t=0
    for (int i=0; i < num_slices; i+=1) {
	    u[i] = 23.0;
    }
    cudaMemcpy(d_u, u, num_slices*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, u, num_slices*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemset(d_u, 23, num_slices * sizeof(float));

    // Launch kernel
    float *history = (float*)malloc((num_steps/10000)*sizeof(float));
    const int block_size = 512;
    const int num_blocks = (num_slices + block_size - 1) / block_size;
    const size_t shared_mem_size = 2 * num_slices * sizeof(float);
    for (int t = 0; t < num_steps; t+=1)
    {
        heat_diffusion<<<num_blocks, block_size, shared_mem_size>>>(d_u, d_u_new, num_slices);
        cudaDeviceSynchronize();
	if ((t % 10000) == 0)
        	cudaMemcpy(&history[t/10000], &d_u[plot_idx], sizeof(float), cudaMemcpyDeviceToHost);
        float *temp = d_u;
        d_u = d_u_new;
        d_u_new = temp;
    }

    cudaMemcpy(u, d_u, num_slices * sizeof(float), cudaMemcpyDeviceToHost);

    // Print temperature at specified location and time
    const float location = 0.7;
    const int index = (int)(location / dx);
    printf("Temperature at location %.2f m at time %.6f s: %.6f\n", location, T_FINAL, u[index]);

    // Plot temperature at point over time 
    printf("Plotting temperature at point %.2f m over time:\n", plot_idx*dx);
    for (int t=0; t < (num_steps/10000); t+=1) {
        printf("%.6f\t%.6f\n", t*TIME_STEP, history[t]);
    }

    // Free memory
    free(u);
    free(u_new);
    free(history);
    cudaFree(d_u);
    cudaFree(d_u_new);

    return 0;
}

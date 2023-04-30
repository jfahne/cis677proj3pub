#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define ROD_LENGTH 1.0
#define TIME_STEP 1e-5
#define T_FINAL 0.1

__global__ void heat_diffusion(float *u, float *u_new, int num_slices)
{
    extern __shared__ float shared_mem[];
    float *u_shared = shared_mem;
    float *u_new_shared = &shared_mem[num_slices];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left_idx = (idx == 0) ? idx : idx - 1;
    int right_idx = (idx == (num_slices - 1)) ? idx : idx + 1;

    if (idx < num_slices)
        u_shared[idx] = u[idx];
    __syncthreads();

    if (idx == 0) {
        u_new_shared[idx] = (u_shared[right_idx] + 100)/2;
        u_new[idx] = u_new_shared[idx];
    } else if (idx == (num_slices - 1)) {
        u_new_shared[idx] = (23 + u_shared[left_idx])/2;
        u_new[idx] = u_new_shared[idx];
    } else if (idx < num_slices) {
        u_new_shared[idx] = (u_shared[right_idx] + u_shared[left_idx])/2;
        u_new[idx] = u_new_shared[idx];
    } else {
        ((void) 0);
    }
}

int main()
{
    const int num_slices = 2500;
    const float dx = ROD_LENGTH / num_slices;
    const int num_steps = T_FINAL / TIME_STEP;

    float *u = (float *)malloc(num_slices * sizeof(float));
    float *u_new = (float *)malloc(num_slices * sizeof(float));
    float *d_u, *d_u_new;

    cudaMalloc((void **)&d_u, num_slices * sizeof(float));
    cudaMalloc((void **)&d_u_new, num_slices * sizeof(float));

    // Initialize temperature at t=0
    u[0] = 100.0;
    for (int i = 1; i < num_slices; i++)
    {
        u[i] = 23.0;
    }

    cudaMemcpy(d_u, u, num_slices * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    float *history = (float*)malloc(num_steps*num_slices*sizeof(float));
    const int block_size = 32;
    const int num_blocks = (num_slices + block_size - 1) / block_size;
    const size_t shared_mem_size = 2 * num_slices * sizeof(float);
    for (int t = 0; t < num_steps; t+=1)
    {
        heat_diffusion<<<num_blocks, block_size, shared_mem_size>>>(d_u, d_u_new, num_slices);
        cudaDeviceSynchronize();
        cudaMemcpy(&history[t*num_slices], d_u, num_slices*sizeof(float), cudaMemcpyDeviceToHost);
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
    const int plot_idx = (int)(0.3/dx);
    printf("Plotting temperature at point %.2f m over time:\n", plot_idx*dx);
    for (int t=0; t < num_steps; t+=1) {
        printf("%.6f\t%.6f\n", t*TIME_STEP, history[t*num_slices + plot_idx]);
    }

    // Free memory
    free(u);
    free(u_new);
    free(history);
    cudaFree(d_u);
    cudaFree(d_u_new);

    return 0;
}
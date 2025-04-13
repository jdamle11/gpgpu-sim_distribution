#include <stdio.h>
#include <cuda_runtime.h>

#define N 256  // Size of the array

// Kernel for the first block (block 0)
__global__ void kernel1(int *global_data) {
    // Allocate shared memory in block 0
    __shared__ int shared_data[N];

    int tid = threadIdx.x;
    
    // Each thread loads data into shared memory
    if (tid < N) {
        shared_data[tid] = tid * 2;  // Sample data (multiplying by 2)
    }
    
    __syncthreads();  // Ensure all threads have written to shared memory
    
    // Store data from shared memory into global memory
    if (tid < N) {
        global_data[tid] = shared_data[tid];
    }
}

// Kernel for the second block (block 1)
__global__ void kernel2(int *global_data) {
    // Allocate shared memory in block 1
    __shared__ int shared_data[N];

    int tid = threadIdx.x;

    // Load data from global memory into shared memory
    if (tid < N) {
        shared_data[tid] = global_data[tid];
    }

    __syncthreads();  // Ensure all threads have loaded into shared memory

    // Modify data in shared memory
    if (tid < N) {
        shared_data[tid] += 1;  // Sample modification (incrementing by 1)
    }

    __syncthreads();

    // Store modified data back into global memory
    if (tid < N) {
        global_data[tid] = shared_data[tid];
    }
}

int main() {
    int *d_global_data;
    int *h_global_data = (int*)malloc(N * sizeof(int));

    // Allocate memory on the device
    cudaMalloc(&d_global_data, N * sizeof(int));

    // Launch the first kernel (block 0)
    kernel1<<<1, N>>>(d_global_data);

    // Launch the second kernel (block 1)
    kernel2<<<1, N>>>(d_global_data);

    // Copy result from device to host
    cudaMemcpy(h_global_data, d_global_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print out the result from global memory
    for (int i = 0; i < N; i++) {
        printf("%d ", h_global_data[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_global_data);
    free(h_global_data);

    return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256  // Number of threads in the block

// Kernel function where threads only access their own shared memory
__global__ void kernel() {
    // Declare shared memory
    __shared__ int shared_data[N];

    int tid = threadIdx.x;

    // Initialize shared memory with thread-specific values
    shared_data[tid] = tid * 2;  // Example: initialize shared memory with thread index multiplied by 2

    __syncthreads();  // Ensure all threads have written to shared memory

    // Each thread reads from its own shared memory location
    int value = shared_data[tid];

    // Modify the value (just a simple operation)
    value += 1;

    // Store the modified value back to shared memory
    shared_data[tid] = value;

    // Output the modified value (for illustration purposes, output only the first 10 values)
    if (tid < 10) {
        printf("Thread %d: value = %d\n", tid, shared_data[tid]);
    }
}

int main() {
    // Launch the kernel with 1 block and N threads
    kernel<<<1, N>>>();

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    return 0;
}

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

// Simulated asymmetric shared memory — one block uses more
__shared__ int shared_block0[512];  // Block 0 uses 512 ints //cluster_shared was here
__shared__ int shared_block1[128];  // Block 1 uses only 128 ints //cluster_shared was here

__global__ void dsm_asymmetry_kernel(int* output) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid == 0) {
        // Block 0: Use a larger shared region
        if (tid < 512)
            shared_block0[tid] = 1000 + tid;  // Fill shared memory with unique values
    } else if (bid == 1) {
        // Block 1: Use smaller shared region, and read from Block 0’s shared memory via DSM
        if (tid < 128) {
            int value = shared_block0[tid * 4];  // Access values every 4 elements
            output[tid] = value;  // Store result from DSM read
        }
    }

    // Optional barrier for correctness
    // __shared__  int counter;
    // __shared__  int sense;

    // __shared__ int local_sense;
    // if (threadIdx.x == 0) {
    //     local_sense = atomicAdd(&counter, 1);
    // }
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     if (local_sense == 2 - 1) {
    //         counter = 0;
    //         atomicExch(&sense, 1);
    //     } else {
    //         while (atomicAdd(&sense, 0) == 0);
    //     }
    // }
    // __syncthreads();
}

int main() {
    const int threadsPerBlock = 256;
    const int numBlocks = 2;

    // Output buffer (only for Block 1)
    int* d_output;
    CHECK(cudaMalloc(&d_output, sizeof(int) * 128));

    // Set cluster-level kernel attributes
    CHECK(cudaFuncSetAttribute(dsm_asymmetry_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CHECK(cudaFuncSetAttribute(dsm_asymmetry_kernel,
        cudaFuncAttributeClusterDimMustBeSet, 1));
    CHECK(cudaFuncSetAttribute(dsm_asymmetry_kernel,
        cudaFuncAttributeRequiredClusterWidth, numBlocks));

    // Launch kernel
    dsm_asymmetry_kernel<<<numBlocks, threadsPerBlock>>>(d_output);
    CHECK(cudaDeviceSynchronize());

    // Read results
    int h_output[128];
    CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    printf("Block 1 read these values from Block 0's shared memory:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_output[%d] = %d\n", i, h_output[i]);
    }

    CHECK(cudaFree(d_output));
    return 0;
}
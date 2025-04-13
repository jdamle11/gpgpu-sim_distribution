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

__shared__  int dsm_array[2][256];  // DSM between 2 blocks //__device__
__device__  int block_counter;      // software barrier counter
__device__  int barrier_sense;      // sense flag

__device__ void my_syncthreads_cluster(int total_blocks_in_cluster) {
    int local_sense;
    printf("KMS!!!!!\n");
    if (threadIdx.x == 0) {
        local_sense = atomicAdd(&block_counter, 1);
    }

    __syncthreads(); // intra-block sync

    if (threadIdx.x == 0) {
        if (local_sense == total_blocks_in_cluster - 1) {
            // Last block resets counter and sets barrier
            block_counter = 0;
            atomicExch(&barrier_sense, 1);
        } else {
            // Spin-wait for barrier sense to be set
            while (atomicAdd(&barrier_sense, 0) == 0);
        }
    }

    __syncthreads(); // wait for all threads in block
}

__global__ void dsm_kernel_manual_cluster_barrier(int* output) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    printf("SATISH!!!!!!!\n");
    // // All threads write into the *other* block's DSM region
    int value = (bid == 0) ? tid + 1000 : tid + 2000;
    dsm_array[(bid + 1) % 2][tid] = value;

    // // Manual cluster-wide barrier
    my_syncthreads_cluster(2);  // assumes exactly 2 blocks per cluster

    // // All threads now read their own DSM region
    int read_value = dsm_array[bid][tid];
    output[bid * blockDim.x + tid] = read_value;
}

int main() {
    const int threadsPerBlock = 256;
    const int blocksPerCluster = 2;
    const int totalThreads = threadsPerBlock * blocksPerCluster;

    // Allocate device output buffer
    int* d_output;
    CHECK(cudaMalloc(&d_output, totalThreads * sizeof(int)));

    // Set cluster-wide behavior
    //CHECK(cudaFuncSetAttribute(dsm_kernel_manual_cluster_barrier,
      //  cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    //CHECK(cudaFuncSetAttribute(dsm_kernel_manual_cluster_barrier,
      //  cudaFuncAttributeClusterDimMustBeSet, 1));
    //CHECK(cudaFuncSetAttribute(dsm_kernel_manual_cluster_barrier,
      //  cudaFuncAttributeRequiredClusterWidth, blocksPerCluster));

    // Launch normally — no cooperative kernel launch
    dsm_kernel_manual_cluster_barrier<<<blocksPerCluster, threadsPerBlock>>>(d_output);
    CHECK(cudaDeviceSynchronize());

    // Copy and print output
    int h_output[totalThreads];
    CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    printf("First few results from Block 0:\n");
    for (int i = 0; i < 5; ++i) {
        printf("Thread %d: %d\n", i, h_output[i]);
    }

    printf("First few results from Block 1:\n");
    for (int i = threadsPerBlock; i < threadsPerBlock + 5; ++i) {
        printf("Thread %d: %d\n", i, h_output[i]);
    }

    CHECK(cudaFree(d_output));
    return 0;
}

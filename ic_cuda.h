#pragma once


/////////////////////////////////////////////////
// Host functions

inline __host__ void checkCudaErrors() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error (%d) %s.\n", (int)err, cudaGetErrorString(err));
    }
}



/////////////////////////////////////////////////
// Device functions

inline __device__ int warpReduceOr(int val) {
    #pragma unroll
    for (int mask = warpSize/2; mask > 0; mask /= 2) 
        val |= __shfl_xor(val, mask);
    return val;
}

inline __device__ int warpReduceSum(int val) {
    #pragma unroll
    for (int mask = warpSize/2; mask > 0; mask /= 2) 
        val += __shfl_xor(val, mask);
    return val;
}

inline __device__ int warpReduceSum(int val) {
    #pragma unroll
    for (int mask = warpSize/2; mask > 0; mask /= 2) 
        val += __shfl_xor(val, mask);
    return val;
}


// icbc.h v1.03
// A High Quality BC1 Encoder in CUDA by Ignacio Castano <castano@gmail.com>.
//
// LICENSE:
//  MIT license at the end of this file.




// Utilities and system includes
#include <cuda_runtime.h>

#include <float.h> // for FLT_MAX

#define NUM_THREADS 64        // Number of threads per block.


typedef unsigned int uint;
typedef unsigned short ushort;


template <class T>
inline __host__ __device__ void swap(T &a, T &b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

/*inline __host__ __device__ int max(int a, int b)
{
    return a > b ? a : b;
}*/

//__constant__ float3 kColorMetric = { 0.2126f, 0.7152f, 0.0722f };
__constant__ float3 kColorMetric = { 1.0f, 1.0f, 1.0f };

inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float a, float3 b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}



////////////////////////////////////////////////////////////////////////////////
// PCA

// Use power method to find the first eigenvector.
// http://www.miislita.com/information-retrieval-tutorial/matrix-tutorial-3-eigenvalues-eigenvectors.html
inline __device__ float3 firstEigenVector(float matrix[6])
{
    // 8 iterations seems to be more than enough.

    float3 v = make_float3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 8; i++)
    {
        float x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
        float y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
        float z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
        float m = max(max(x, y), z);
        float iv = 1.0f / m;
        v = make_float3(x*iv, y*iv, z*iv);
    }

    return v;
}

inline __device__ void colorSums(const float3 *colors, float3 *sums)
{
    const int idx = threadIdx.x;

    sums[idx] = colors[idx];
    sums[idx] += sums[idx^8];
    sums[idx] += sums[idx^4];
    sums[idx] += sums[idx^2];
    sums[idx] += sums[idx^1];

    // @@ Use warp-wide reduction.
}


inline __device__ float3 bestFitLine(const float3 *colors, float3 color_sum)
{
    // Compute covariance matrix of the given colors.
    const int idx = threadIdx.x;

    float3 diff = colors[idx] - color_sum * (1.0f / 16.0f);

    // @@ Eliminate two-way bank conflicts here.
    // @@ It seems that doing that and unrolling the reduction doesn't help...
    __shared__ float covariance[16*6];

    covariance[6 * idx + 0] = diff.x * diff.x;    // 0, 6, 12, 2, 8, 14, 4, 10, 0
    covariance[6 * idx + 1] = diff.x * diff.y;
    covariance[6 * idx + 2] = diff.x * diff.z;
    covariance[6 * idx + 3] = diff.y * diff.y;
    covariance[6 * idx + 4] = diff.y * diff.z;
    covariance[6 * idx + 5] = diff.z * diff.z;

    for (int d = 8; d > 0; d >>= 1)
    {
        if (idx < d)
        {
            covariance[6 * idx + 0] += covariance[6 * (idx+d) + 0];
            covariance[6 * idx + 1] += covariance[6 * (idx+d) + 1];
            covariance[6 * idx + 2] += covariance[6 * (idx+d) + 2];
            covariance[6 * idx + 3] += covariance[6 * (idx+d) + 3];
            covariance[6 * idx + 4] += covariance[6 * (idx+d) + 4];
            covariance[6 * idx + 5] += covariance[6 * (idx+d) + 5];
        }
        }

    // Compute first eigen vector.
    return firstEigenVector(covariance);
}



////////////////////////////////////////////////////////////////////////////////
// Compute Permutations

__device__ uint d_permutations[1024];

__host__ static void computePermutations(uint permutations[1024])
{
    int indices[16];
    int num = 0;

    // 3 element permutations:

    // first cluster [0,i) is at the start
    for (int m = 0; m < 16; ++m)
    {
        indices[m] = 0;
    }

    const int imax = 15;

    for (int i = imax; i >= 0; --i)
    {
        // second cluster [i,j) is half along
        for (int m = i; m < 16; ++m)
        {
            indices[m] = 2;
        }

        const int jmax = (i == 0) ? 15 : 16;

        for (int j = jmax; j >= i; --j)
        {
            // last cluster [j,k) is at the end
            if (j < 16)
            {
                indices[j] = 1;
            }

            uint permutation = 0;

            for (int p = 0; p < 16; p++)
            {
                permutation |= indices[p] << (p * 2);
                //permutation |= indices[15-p] << (p * 2);
            }

            permutations[num] = permutation;

            num++;
        }
    }

    assert(num == 151);

    for (int i = 0; i < 9; i++)
    {
        permutations[num] = 0x000AA555;
        num++;
    }

    assert(num == 160);

    // Append 4 element permutations:

    // first cluster [0,i) is at the start
    for (int m = 0; m < 16; ++m)
    {
        indices[m] = 0;
    }

    for (int i = imax; i >= 0; --i)
    {
        // second cluster [i,j) is one third along
        for (int m = i; m < 16; ++m)
        {
            indices[m] = 2;
        }

        const int jmax = (i == 0) ? 15 : 16;

        for (int j = jmax; j >= i; --j)
        {
            // third cluster [j,k) is two thirds along
            for (int m = j; m < 16; ++m)
            {
                indices[m] = 3;
            }

            int kmax = (j == 0) ? 15 : 16;

            for (int k = kmax; k >= j; --k)
            {
                // last cluster [k,n) is at the end
                if (k < 16)
                {
                    indices[k] = 1;
                }

                uint permutation = 0;

                bool hasThree = false;

                for (int p = 0; p < 16; p++)
                {
                    permutation |= indices[p] << (p * 2);
                    //permutation |= indices[15-p] << (p * 2);

                    if (indices[p] == 3) hasThree = true;
                }

                if (hasThree)
                {
                    permutations[num] = permutation;
                    num++;
                }
            }
        }
    }

    assert(num == 975);

    // 1024 - 969 - 7 = 48 extra elements

    // It would be nice to set these extra elements with better values...
    for (int i = 0; i < 49; i++)
    {
        permutations[num] = 0x00AAFF55;
        num++;
    }

    assert(num == 1024);
}

__host__ void compute_permutations() {
    // Compute permutations.
    uint permutations[1024];
    computePermutations(permutations);

    // Copy permutations host to devie.
    cudaMemcpyToSymbol(d_permutations, permutations, 1024 * sizeof(uint));

}




////////////////////////////////////////////////////////////////////////////////
// Sort colors
////////////////////////////////////////////////////////////////////////////////
__device__ void sortColors(const float *values, int *ranks)
{
    const int tid = threadIdx.x;

    int rank = 0;

#pragma unroll

    for (int i = 0; i < 16; i++)
    {
        rank += (values[i] < values[tid]);
    }

    ranks[tid] = rank;

    // Resolve elements with the same index.
    for (int i = 0; i < 15; i++)
    {
        if (tid > i && ranks[tid] == ranks[i])
        {
            ++ranks[tid];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Load color block to shared mem
////////////////////////////////////////////////////////////////////////////////
__device__ void loadColorBlock(const uint *image, float3 colors[16], float3 sums[16], int xrefs[16], int blockOffset)
{
    const int bid = blockIdx.x + blockOffset;
    const int idx = threadIdx.x;

    __shared__ float dps[16];

    float3 tmp;

    if (idx < 16)
    {
        // Read color and copy to shared mem.
        uint c = image[(bid) * 16 + idx];

        colors[idx].x = ((c >> 0) & 0xFF) * (1.0f / 255.0f);
        colors[idx].y = ((c >> 8) & 0xFF) * (1.0f / 255.0f);
        colors[idx].z = ((c >> 16) & 0xFF) * (1.0f / 255.0f);

        // Sort colors along the best fit line.
        colorSums(colors, sums);

        float3 axis = bestFitLine(colors, sums[0]);

        dps[idx] = dot(colors[idx], axis);

        sortColors(dps, xrefs);

        tmp = colors[idx];

        colors[xrefs[idx]] = tmp;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Round color to RGB565 and expand
////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 roundAndExpand(float3 v, ushort *w)
{
    v.x = rintf(__saturatef(v.x) * 31.0f);
    v.y = rintf(__saturatef(v.y) * 63.0f);
    v.z = rintf(__saturatef(v.z) * 31.0f);

    *w = ((ushort)v.x << 11) | ((ushort)v.y << 5) | (ushort)v.z;
    v.x *= 0.03227752766457f; // approximate integer bit expansion.
    v.y *= 0.01583151765563f;
    v.z *= 0.03227752766457f;
    return v;
}


__constant__ float alphaTable4[4] = { 9.0f, 0.0f, 6.0f, 3.0f };
__constant__ float alphaTable3[4] = { 4.0f, 0.0f, 2.0f, 2.0f };
__constant__ const int prods4[4] = { 0x090000,0x000900,0x040102,0x010402 };
__constant__ const int prods3[4] = { 0x040000,0x000400,0x040101,0x010401 };

#define USE_TABLES 1

////////////////////////////////////////////////////////////////////////////////
// Evaluate permutations
////////////////////////////////////////////////////////////////////////////////
static __device__ float evalPermutation4(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
    // Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable4[bits & 3] * colors[i];
        akku += prods4[bits & 3];
    }

    float alpha2_sum = float(akku >> 16);
    float beta2_sum = float((akku >> 8) & 0xff);
    float alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum = (9.0f * color_sum) - alphax_sum;
#else
    float alpha2_sum = 0.0f;
    float beta2_sum = 0.0f;
    float alphabeta_sum = 0.0f;
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        float beta = (bits & 1);

        if (bits & 2)
        {
            beta = (1 + beta) * (1.0f / 3.0f);
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    // alpha2, beta2, alphabeta and factor could be precomputed for each permutation, but it's faster to recompute them.
    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.111111111111f) * dot(e, kColorMetric);
}

static __device__ float evalPermutation3(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
    // Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable3[bits & 3] * colors[i];
        akku += prods3[bits & 3];
    }

    float alpha2_sum = float(akku >> 16);
    float beta2_sum = float((akku >> 8) & 0xff);
    float alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum = (4.0f * color_sum) - alphax_sum;
#else
    float alpha2_sum = 0.0f;
    float beta2_sum = 0.0f;
    float alphabeta_sum = 0.0f;
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        float beta = (bits & 1);

        if (bits & 2)
        {
            beta = 0.5f;
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.25f) * dot(e, kColorMetric);
}

__device__ void evalAllPermutations(const float3 *colors, ushort &bestStart, ushort &bestEnd, uint &bestPermutation, float *errors, float3 color_sum)
{
    const int idx = threadIdx.x;

    float bestError = FLT_MAX;

    __shared__ uint s_permutations[160];

    for (int i = 0; i < 16; i++)
    {
        int pidx = idx + NUM_THREADS * i;

        if (pidx >= 992)
        {
            break;
        }

        ushort start, end;
        uint permutation = d_permutations[pidx];

        if (pidx < 160)
        {
            s_permutations[pidx] = permutation;
        }

        float error = evalPermutation4(colors, permutation, &start, &end, color_sum);

        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;
        }
    }

    if (bestStart < bestEnd)
    {
        swap(bestEnd, bestStart);
        bestPermutation ^= 0x55555555;    // Flip indices.
    }

    // Sync here to ensure s_permutations is valid going forward
    __syncthreads();

    for (int i = 0; i < 3; i++)
    {
        int pidx = idx + NUM_THREADS * i;

        if (pidx >= 160)
        {
            break;
        }

        ushort start, end;
        uint permutation = s_permutations[pidx];
        float error = evalPermutation3(colors, permutation, &start, &end, color_sum);

        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;

            if (bestStart > bestEnd)
            {
                swap(bestEnd, bestStart);
                bestPermutation ^= (~bestPermutation >> 1) & 0x55555555;    // Flip indices.
            }
        }
    }

    errors[idx] = bestError;
}

////////////////////////////////////////////////////////////////////////////////
// Find index with minimum error
////////////////////////////////////////////////////////////////////////////////
__device__ int findMinError(float *errors)
{
    const int idx = threadIdx.x;
    __shared__ int indices[NUM_THREADS];
    indices[idx] = idx;

    __syncthreads();

    for (int d = NUM_THREADS/2; d > 0; d >>= 1)
    {
        float err0 = errors[idx];
        float err1 = (idx + d) < NUM_THREADS ? errors[idx + d] : FLT_MAX;
        int index1 = (idx + d) < NUM_THREADS ? indices[idx + d] : 0;

        __syncthreads();

        if (err1 < err0)
        {
            errors[idx] = err1;
            indices[idx] = index1;
        }

        __syncthreads();
    }

    return indices[0];
}

////////////////////////////////////////////////////////////////////////////////
// Save DXT block
////////////////////////////////////////////////////////////////////////////////
__device__ void saveBlockDXT1(ushort start, ushort end, uint permutation, int xrefs[16], uint2 *result, int blockOffset)
{
    const int bid = blockIdx.x + blockOffset;

    if (start == end)
    {
        permutation = 0;
    }

    // Reorder permutation.
    uint indices = 0;

    for (int i = 0; i < 16; i++)
    {
        int ref = xrefs[i];
        indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
    }

    // Write endpoints.
    result[bid].x = (end << 16) | start;

    // Write palette indices.
    result[bid].y = indices;
}




////////////////////////////////////////////////////////////////////////////////
// Compress color block
////////////////////////////////////////////////////////////////////////////////
__global__ void compress_dxt1(const uint *image, uint2 *result, int blockOffset)
{
    const int idx = threadIdx.x;

    __shared__ float3 colors[16];
    __shared__ float3 sums[16];
    __shared__ int xrefs[16];

    loadColorBlock(image, colors, sums, xrefs, blockOffset);

    __syncthreads();

    ushort bestStart, bestEnd;
    uint bestPermutation;

    __shared__ float errors[NUM_THREADS];

    evalAllPermutations(colors, bestStart, bestEnd, bestPermutation, errors, sums[0]);

    // Use a parallel reduction to find minimum error.
    const int minIdx = findMinError(errors);

    __syncthreads();

    // Only write the result of the winner thread.
    if (idx == minIdx)
    {
        saveBlockDXT1(bestStart, bestEnd, bestPermutation, xrefs, result, blockOffset);
    }
}











#if 0


#ifndef ICBC_CUH
#define ICBC_CUH

namespace icbc {

    void cuda_compress_dxt1(const float color_weights[3], bool three_color_mode, bool three_color_black, void * output);

}

#endif // ICBC_CUH


#ifdef ICBC_CU_IMPLEMENTATION

__device__ void saveBlockDXT1_Parallel(uint endpoints, float3 colors[16], int xrefs[16], uint * result)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid < 16)
    {
        int3 color = float3ToInt3(colors[xrefs[tid]]);

        ushort endpoint0 = endpoints & 0xFFFF;
        ushort endpoint1 = endpoints >> 16;

        int3 palette[4];
        palette[0] = color16ToInt3(endpoint0);
        palette[1] = color16ToInt3(endpoint1);

        int d0 = colorDistance(palette[0], color);
        int d1 = colorDistance(palette[1], color);

        uint index;
        if (endpoint0 > endpoint1) 
        {
            palette[2].x = (2 * palette[0].x + palette[1].x) / 3;
            palette[2].y = (2 * palette[0].y + palette[1].y) / 3;
            palette[2].z = (2 * palette[0].z + palette[1].z) / 3;

            palette[3].x = (2 * palette[1].x + palette[0].x) / 3;
            palette[3].y = (2 * palette[1].y + palette[0].y) / 3;
            palette[3].z = (2 * palette[1].z + palette[0].z) / 3;
            
            int d2 = colorDistance(palette[2], color);
            int d3 = colorDistance(palette[3], color);

            // Compute the index that best fit color.
            uint b0 = d0 > d3;
            uint b1 = d1 > d2;
            uint b2 = d0 > d2;
            uint b3 = d1 > d3;
            uint b4 = d2 > d3;

            uint x0 = b1 & b2;
            uint x1 = b0 & b3;
            uint x2 = b0 & b4;

            index = (x2 | ((x0 | x1) << 1));
        }
        else {
            palette[2].x = (palette[0].x + palette[1].x) / 2;
            palette[2].y = (palette[0].y + palette[1].y) / 2;
            palette[2].z = (palette[0].z + palette[1].z) / 2;

            int d2 = colorDistance(palette[2], color);

            index = 0;
            if (d1 < d0 && d1 < d2) index = 1;
            else if (d2 < d0) index = 2;
        }

        __shared__ uint indices[16];

        indices[tid] = index << (2 * tid);
        if (tid < 8) indices[tid] |= indices[tid+8];
        if (tid < 4) indices[tid] |= indices[tid+4];
        if (tid < 2) indices[tid] |= indices[tid+2];
        if (tid < 1) indices[tid] |= indices[tid+1];

        if (tid < 2) {
            result[2 * bid + tid] = tid == 0 ? endpoints : indices[0];
        }
    }
}


__global__ void compressDXT1(uint firstBlock, uint blockWidth, const uint * permutations, uint2 * result)
{
    __shared__ float3 colors[16];
    __shared__ float3 sums[16];
    __shared__ int xrefs[16];
    __shared__ int sameColor;

    loadColorBlockTex(firstBlock, blockWidth, colors, sums, xrefs, &sameColor);

    __syncthreads();

    if (sameColor)
    {
        if (threadIdx.x == 0) saveSingleColorBlockDXT1(colors[0], result);
        return;
    }

    ushort bestStart, bestEnd;
    uint bestPermutation;

    __shared__ float errors[NUM_THREADS];
    evalAllPermutations(colors, sums[0], permutations, bestStart, bestEnd, bestPermutation, errors);
    
    // Use a parallel reduction to find minimum error.
    const int minIdx = findMinError(errors);

    __shared__ uint s_bestEndPoints;

    // Only write the result of the winner thread.
    if (threadIdx.x == minIdx)
    {
        s_bestEndPoints = (bestEnd << 16) | bestStart;
    }

    __syncthreads();

    saveBlockDXT1_Parallel(s_bestEndPoints, colors, xrefs, (uint *)result);
}


namespace icbc {

    void cuda_compress_dxt1(const float color_weights[3], bool three_color_mode, bool three_color_black, void * output) {
    
    }

}

#endif // ICBC_CU_IMPLEMENTATION

#endif // 0


/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

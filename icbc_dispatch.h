// icbc_dispatch.h v1.04
// A dynamic CPU dispatch for ICBC by Ignacio Castano <castano@gmail.com>.
//
// LICENSE:
//  MIT license at the end of this file.

#ifndef ICBC_DISPATCH_H
#define ICBC_DISPATCH_H

#define ICBC_X86 1
//#define ICBC_X64    (defined(__x86_64__) || defined(_M_X64))
//#define ICBC_X86    !ICBC_X64 && (defined(__i386__) || defined(_M_IX86))
//#define ICBC_ARM

namespace icbc {

    void init_dxt1();

    enum Quality {
        Quality_Level1,  // Box fit + least squares fit.
        Quality_Level2,  // Cluster fit 4, threshold = 24.
        Quality_Level3,  // Cluster fit 4, threshold = 32.
        Quality_Level4,  // Cluster fit 4, threshold = 48.
        Quality_Level5,  // Cluster fit 4, threshold = 64.
        Quality_Level6,  // Cluster fit 4, threshold = 96.
        Quality_Level7,  // Cluster fit 4, threshold = 128.
        Quality_Level8,  // Cluster fit 4+3, threshold = 256.
        Quality_Level9,  // Cluster fit 4+3, threshold = 256 + Refinement.

        Quality_Fast = Quality_Level1,
        Quality_Default = Quality_Level8,
        Quality_Max = Quality_Level9,
    };

    float compress_dxt1(Quality level, const float * input_colors, const float * input_weights, const float color_weights[3], bool three_color_mode, bool three_color_black, void * output);

    enum Decoder {
        Decoder_D3D10 = 0,
        Decoder_NVIDIA = 1,
        Decoder_AMD = 2
    };

    void decode_dxt1(const void * block, unsigned char rgba_block[16 * 4], Decoder decoder = Decoder_D3D10);
    float evaluate_dxt1_error(const unsigned char rgba_block[16 * 4], const void * block, Decoder decoder = Decoder_D3D10);
}

#endif // ICBC_DISPATCH_H

#ifdef ICBC_IMPLEMENTATION

#undef ICBC_SIMD
#define ICBC_SIMD 0
#define icbc icbc_float
#include "icbc.h"
#undef ICBC_H
#undef icbc

// If x86:
//#if ICBC_X86 || ICBC_X64
    #define ICBC_SIMD ICBC_SSE2
    #define icbc icbc_sse2
    #include "icbc.h"
    #undef ICBC_H
    #undef icbc

    #define ICBC_SIMD ICBC_SSE41
    #define icbc icbc_sse41
    #include "icbc.h"
    #undef ICBC_H
    #undef icbc

    #define ICBC_SIMD ICBC_AVX1
    #define icbc icbc_avx1
    #include "icbc.h"
    #undef ICBC_H
    #undef icbc

    #define ICBC_SIMD ICBC_AVX2
    #define icbc icbc_avx2
    #include "icbc.h"
    #undef ICBC_H
    #undef icbc

    #define ICBC_SIMD ICBC_AVX512
    #define icbc icbc_avx512
    #include "icbc.h"
    #undef ICBC_H
    #undef icbc
//#endif

// If ARM:
/*#if ICBC_ARM
    #define ICBC_SIMD -1
    #define icbc icbc_neon
    #include "icbc.h"
#endif*/

#undef icbc

namespace icbc {

static int simd_version = -1;


// https://software.intel.com/content/www/us/en/develop/articles/how-to-detect-new-instruction-support-in-the-4th-generation-intel-core-processor-family.html
inline void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
{
#if defined(_MSC_VER)
    __cpuidex(abcd, eax, ecx);
#else
    uint32_t ebx, edx;
# if defined( __i386__ ) && defined ( __PIC__ )
     /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__ ( "movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D" (ebx),
# else
    __asm__ ( "cpuid" : "+b" (ebx),
# endif
              "+a" (eax), "+c" (ecx), "=d" (edx) );
    abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
#endif
}

inline int check_xcr0_ymm()
{
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);  /* min VS2010 SP1 compiler is required */
#else
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif
    return ((xcr0 & 6) == 6); /* checking if xmm and ymm state are enabled in XCR0 */
}

// check_4th_gen_intel_core_features
static bool check_avx2() {
    uint32_t abcd[4];
    uint32_t fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);

    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1   && 
       CPUID.(EAX=01H, ECX=0H):ECX.MOVBE[bit 22]==1 && 
       CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 */
    run_cpuid( 1, 0, abcd );
    if ( (abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask ) 
        return false;

    if (!check_xcr0_ymm())
        return false;

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    run_cpuid( 7, 0, abcd );
    if ( (abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask ) 
        return false;

    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    run_cpuid( 0x80000001, 0, abcd );
    if ( (abcd[2] & (1 << 5)) == 0)
        return false;

    return true;
}

inline void detect_simd_version() {
    if (simd_version < 0) {
        simd_version = ICBC_FLOAT;

        // Check for SSE2
        //simd_version = ICBC_SSE2;

        // Check for SSE41
        //simd_version = ICBC_SSE41;

        // Check for AVX
        //simd_version = ICBC_AVX1;

        // Check for AVX2
        if (check_avx2())
            simd_version = ICBC_AVX2;

        // Check for AVX512F
        //simd_version = ICBC_AVX512;

        // @@ Use cpuid to determine processor.
        // @@ Check for bmi2 also.
        // @@ Make sure AVX is supported by the OS.
        simd_version = ICBC_AVX512;
    }
}


void init_dxt1() {
    
    detect_simd_version();

    switch(simd_version) {
    #if ICBC_X86
        case ICBC_AVX512:
            return icbc_avx512::init_dxt1();
        case ICBC_AVX2:
            return icbc_avx2::init_dxt1();
        case ICBC_AVX1:
            return icbc_avx1::init_dxt1();
        case ICBC_SSE41:
            return icbc_sse41::init_dxt1();
        case ICBC_SSE2:
            return icbc_sse2::init_dxt1();
    #endif // ICBC_X86
        default:
            return icbc_float::init_dxt1();
    };
}


float compress_dxt1(Quality level, const float * input_colors, const float * input_weights, const float color_weights[3], bool three_color_mode, bool three_color_black, void * output) {
    ICBC_ASSERT(simd_version >= 0);

    switch(simd_version) {
    #if ICBC_X86
        case ICBC_AVX512:
            return icbc_avx512::compress_dxt1((icbc_avx512::Quality)level, input_colors, input_weights, color_weights, three_color_mode, three_color_black, output);
        case ICBC_AVX2:
            return icbc_avx2::compress_dxt1((icbc_avx2::Quality)level, input_colors, input_weights, color_weights, three_color_mode, three_color_black, output);
        case ICBC_AVX1:
            return icbc_avx1::compress_dxt1((icbc_avx1::Quality)level, input_colors, input_weights, color_weights, three_color_mode, three_color_black, output);
        case ICBC_SSE41:
            return icbc_sse41::compress_dxt1((icbc_sse41::Quality)level, input_colors, input_weights, color_weights, three_color_mode, three_color_black, output);
        case ICBC_SSE2:
            return icbc_sse2::compress_dxt1((icbc_sse2::Quality)level, input_colors, input_weights, color_weights, three_color_mode, three_color_black, output);
    #endif // ICBC_X86
        default:
            return icbc_float::compress_dxt1((icbc_float::Quality)level, input_colors, input_weights, color_weights, three_color_mode, three_color_black, output);
    };
}

void decode_dxt1(const void * block, unsigned char rgba_block[16 * 4], Decoder decoder) {
    icbc_float::decode_dxt1(block, rgba_block, (icbc_float::Decoder)decoder);
}

float evaluate_dxt1_error(const unsigned char rgba_block[16 * 4], const void * block, Decoder decoder) {
    return icbc_float::evaluate_dxt1_error(rgba_block, block, (icbc_float::Decoder)decoder);
}

} // icbc namespace

#define ICBC_H
#endif // ICBC_IMPLEMENTATION

// Version History:
// v1.04 - Initial release.

// Copyright (c) 2020 Ignacio Castano <castano@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to  deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

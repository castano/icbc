// Compilation instructions:
// $ g++ icbc_test.cpp -O3 -mavx2 -lpthread -std=c+=11
// > cl icbc_test.cpp /O2 /arch:AVX2

// Enable one of these:
//#define ICBC_SIMD 0         // FLOAT
//#define ICBC_SIMD 1         // SSE2
//#define ICBC_SIMD 2         // SSE4.1
//#define ICBC_SIMD 3         // AVX
#define ICBC_SIMD 4         // AVX2
//#define ICBC_SIMD 5         // AVX512
//#define ICBC_SIMD -1        // NEON
//#define ICBC_SIMD -2        // VMX
//#define ICBC_SIMD -3        // WASM

//#define TRACY_ENABLE
#include "tracy/Tracy.hpp"

#define ICBC_IMPLEMENTATION
#include "icbc.h"

// stb_image from: https://github.com/nothings/stb/blob/master/stb_image.h
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define IC_PFOR_IMPLEMENTATION
#include "ic_pfor.h"



#include <stdio.h>
#include <stdint.h>


////////////////////////////////
// Basic types

typedef unsigned char u8;
typedef unsigned int u32;
typedef uint64_t u64;


////////////////////////////////
// defer

#define CONCAT_INTERNAL(x,y) x##y
#define CONCAT(x,y) CONCAT_INTERNAL(x,y)

template<typename T>
struct ExitScope {
    T lambda;
    ExitScope(T lambda):lambda(lambda){}
    ~ExitScope(){lambda();}
private:
    ExitScope& operator=(const ExitScope&);
};

struct ExitScopeHelp {
    template<typename T>
        ExitScope<T> operator+(T t){ return t;}
};

#if _MSC_VER
#define defer const auto& CONCAT(defer__, __LINE__) = ExitScopeHelp() + [&]()
#else // __GNUC__ or __clang__
#define defer const auto& __attribute__((unused)) CONCAT(defer__, __LINE__) = ExitScopeHelp() + [&]()
#endif


////////////////////////////////
// Timer

// Based of https://gist.github.com/pervognsen/496d659251ad2af200dde4c773bd565f
#if defined _WIN32 || __CYGWIN__
#include <windows.h>
inline double timer_frequency() {
    LARGE_INTEGER qpf;
    QueryPerformanceFrequency(&qpf);
    return double(qpf.QuadPart);
}
inline u64 timer_time() {
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    return qpc.QuadPart;
}
#elif defined(__APPLE__) || defined(__MACH__)
#include <mach/mach_time.h>
inline double timer_frequency() {
    mach_timebase_info_data_t mach_timebase;
    mach_timebase_info(&mach_timebase);
    return double(mach_timebase.denom) * 1e9;
}
inline u64 timer_time() {
    return mach_absolute_time();
}
#else
#include <time.h>
inline double timer_frequency() {
    return 1e9;
}
inline u64 timer_time() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return -1;
    }
    return u64(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}
#endif

struct Timer {
    double freq;
    u64 time_start;

    Timer() {
        freq = timer_frequency();
        time_start = 0;
    }

    void start() {
        time_start = timer_time();
    }

    double secs() {
        if (time_start == 0) return 0.0;
        u64 time_now = timer_time();
        return (time_now - time_start) / freq;
    }

    double stop() {
        double t = secs();
        time_start = 0;
        return t;
    }
};

struct TimeEstimate {
    int num = 0;        // number of samples
    double unit = 1.0;  // units per second
    double min = 0.0;   // min sample
    double max = 0.0;   // max sample
    double avg = 0.0;   // average sample

    void add(double sample) {
        num++;
        min = (num == 1 || sample < min) ? sample : min;
        max = (num == 1 || sample > max) ? sample : max;
        avg += (sample - avg) / num;
    }
};


////////////////////////////////
// File Output

#define IC_MAKEFOURCC(str) (u32(str[0]) | (u32(str[1]) << 8) | (u32(str[2]) << 16) | (u32(str[3]) << 24 ))

static bool output_dxt_dds(u32 w, u32 h, const u8* data, const char * filename) {

    const u32 DDSD_CAPS = 0x00000001;
    const u32 DDSD_PIXELFORMAT = 0x00001000;
    const u32 DDSD_WIDTH = 0x00000004;
    const u32 DDSD_HEIGHT = 0x00000002;
    const u32 DDSD_LINEARSIZE = 0x00080000;
    const u32 DDPF_FOURCC = 0x00000004;
    const u32 DDSCAPS_TEXTURE = 0x00001000;

    struct DDS {
        u32 fourcc = IC_MAKEFOURCC("DDS ");
        u32 size = 124;
        u32 flags = DDSD_CAPS|DDSD_PIXELFORMAT|DDSD_WIDTH|DDSD_HEIGHT|DDSD_LINEARSIZE;
        u32 height = 0;
        u32 width = 0;
        u32 pitch = 0;
        u32 depth = 0;
        u32 mipmapcount = 1;
        u32 reserved [11];
        struct {
            u32 size = 32;
            u32 flags = DDPF_FOURCC;
            u32 fourcc = IC_MAKEFOURCC("DXT1");
            u32 bitcount = 0;
            u32 rmask = 0;
            u32 gmask = 0;
            u32 bmask = 0;
            u32 amask = 0;
        } pf;
        struct {
            u32 caps1 = DDSCAPS_TEXTURE;
            u32 caps2 = 0;
            u32 caps3 = 0;
            u32 caps4 = 0;
        } caps;
        u32 notused = 0;
    } dds;
    static_assert(sizeof(DDS) == 128, "DDS size must be 128");

    dds.width = w;
    dds.height = h;
    dds.pitch = 8 * (((w+3)/4) * ((h+3)/4)); // linear size

    FILE * fp = fopen(filename, "wb");
    if (fp == nullptr) return false;

    // Write header:
    fwrite(&dds, sizeof(dds), 1, fp);

    // Write dxt data:
    fwrite(data, dds.pitch, 1, fp);

    fclose(fp);

    return true;
}

static bool output_dxt_ktx(u32 w, u32 h, const u8* data, const char * filename) {

    const u32 GL_COMPRESSED_RGB_S3TC_DXT1_EXT = 0x83F0;
    const u32 GL_RGBA = 0x1908;

    struct KTX {
        // '«', 'K', 'T', 'X', ' ', '2', '0', '»', '\r', '\n', '\x1A', '\n'
        u8 identifier[12] = { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
        u32 endianness = 0x04030201;
        u32 glType = 0;
        u32 glTypeSize = 0;
        u32 glFormat = 0;
        u32 glInternalFormat = 0;
        u32 glBaseInternalFormat = 0;
        u32 pixelWidth = 0;
        u32 pixelHeight = 0;
        u32 pixelDepth = 0;
        u32 numberOfArrayElements = 0;
        u32 numberOfFaces = 0;
        u32 numberOfMipmapLevels = 0;
        u32 bytesOfKeyValueData = 0;
    } ktx;

    ktx.glTypeSize = 1;
    ktx.glInternalFormat = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
    ktx.glBaseInternalFormat = GL_RGBA;
    ktx.pixelWidth = w;
    ktx.pixelHeight = h;
    ktx.numberOfFaces = 1;
    ktx.numberOfMipmapLevels = 1;

    u32 image_size = 8 * ((w+3)/4 * (h+3)/4);

    FILE * fp = fopen(filename, "wb");
    if (fp == nullptr) return false;

    // Write header:
    fwrite(&ktx, sizeof(ktx), 1, fp);
    fwrite(&image_size, sizeof(u32), 1, fp);

    // Write dxt data:
    fwrite(data, image_size, 1, fp);

    fclose(fp);

    return true;
}

static bool output_dxt_png(u32 w, u32 h, const u8* data, const char * filename) {

    FILE * fp = fopen(filename, "wb");
    if (fp == nullptr) return false;

    // Write header:
    //fwrite(&ktx, sizeof(ktx), 1, fp);
    //fwrite(&image_size, sizeof(u32), 1, fp);

    // Write dxt data:
    //fwrite(data, image_size, 1, fp);

    fclose(fp);

    return true;
}


////////////////////////////////
// DXT

// Returns mse.
static float evaluate_dxt1_mse(u8 * rgba, u8 * block, int block_count, icbc::Decoder decoder = icbc::Decoder_D3D10) {
    double total = 0.0f;
    for (int b = 0; b < block_count; b++) {
        total += icbc::evaluate_dxt1_error(rgba, block, decoder);
        rgba += 4 * 4 * 4;
        block += 8;
    }
    return float(total / (16 * block_count));
}


static float mse_to_psnr(float mse) {
    float rms = sqrtf(mse);
    float psnr = rms ? (float)icbc::clamp(log10(255.0 / rms) * 20.0, 0.0, 300.0) : 1e+10f;
    return psnr;
}


// Input options:
bool output_dds = false;
bool output_ktx = false;
bool output_png = false;
int repeat_count = 1;
icbc::Quality quality_level = icbc::Quality_Default;

// Output stats:
int total_block_count = 0;
double total_avg_time = 0;
double total_min_time = 0;
double total_mse = 0;

bool encode_image(const char * input_filename) {

    int w, h, n;
    unsigned char *input_data = stbi_load(input_filename, &w, &h, &n, 4);
    defer { stbi_image_free(input_data); };

    if (input_data == nullptr) {
        printf("Failed to load input image '%s'.\n", input_filename);
        return false;
    }

    int block_count = (w / 4) * (h / 4);
    u8 * rgba_block_data = (u8 *)malloc(block_count * 4 * 4 * 4);
    defer { free(rgba_block_data); };

    int bw = 4 * (w / 4); // @@ Round down.
    int bh = 4 * (h / 4);

    // Convert to block layout.
    for (int y = 0, b = 0; y < bh; y += 4) {
        for (int x = 0; x < bw; x += 4, b++) {
            for (int yy = 0; yy < 4; yy++) {
                for (int xx = 0; xx < 4; xx++) {
                    if (x + xx < w && y + yy < h) {
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 0] = input_data[((y + yy) * w + x + xx) * 4 + 0];
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 1] = input_data[((y + yy) * w + x + xx) * 4 + 1];
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 2] = input_data[((y + yy) * w + x + xx) * 4 + 2];
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 3] = input_data[((y + yy) * w + x + xx) * 4 + 3];
                    }
                    else {
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 0] = 0;
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 1] = 0;
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 2] = 0;
                        rgba_block_data[b * 4 * 4 * 4 + (yy * 4 + xx) * 4 + 3] = 0;
                    }
                }
            }
        }
    }

    //const float color_weights[3] = {3, 4, 2}; // This is probably better for color images.
    const float color_weights[3] = {1, 1, 1};

    u8 * block_data = (u8 *)malloc(block_count * 8);

    printf("Encoding '%s':", input_filename);

    TimeEstimate estimate;
    Timer timer;
    for (int i = 0; i < repeat_count; i++) {

        timer.start();

        //for (int b = 0; b < block_count; b++) {
        //ic::pfor(block_count, 32, [=](int b) {
        ic_pfor(b, block_count, 32) {
            float input_colors[16 * 4];
            float input_weights[16];
            for (int j = 0; j < 16; j++) {
                input_colors[4 * j + 0] = rgba_block_data[b * 4 * 4 * 4 + j * 4 + 0] / 255.0f;
                input_colors[4 * j + 1] = rgba_block_data[b * 4 * 4 * 4 + j * 4 + 1] / 255.0f;
                input_colors[4 * j + 2] = rgba_block_data[b * 4 * 4 * 4 + j * 4 + 2] / 255.0f;
                input_colors[4 * j + 3] = 1.0f;
                input_weights[j] = 1.0f;
            }

            icbc::compress_dxt1(quality_level, input_colors, input_weights, color_weights, /*three_color_mode=*/true, /*three_color_black=*/true, (block_data + b * 8));
        };
        //});

        estimate.add(timer.stop());
    }

    float mse = evaluate_dxt1_mse(rgba_block_data, block_data, block_count);

    char output_filename[1024];
    if (output_dds) {
        snprintf(output_filename, 1024, "%.*s_bc1.dds", int(strchr(input_filename, '.')-input_filename), input_filename);
        output_dxt_dds(bw, bh, block_data, output_filename);
    }
    if (output_ktx) {
        snprintf(output_filename, 1024, "%.*s_bc1.ktx", int(strchr(input_filename, '.')-input_filename), input_filename);
        output_dxt_ktx(bw, bh, block_data, output_filename);
    }
    if (output_png) {
        snprintf(output_filename, 1024, "%.*s_bc1.png", int(strchr(input_filename, '.')-input_filename), input_filename);
        output_dxt_png(bw, bh, block_data, output_filename);
    }

    total_block_count += block_count;
    total_avg_time += estimate.avg;
    total_min_time += estimate.min;
    total_mse += mse * block_count;

    printf("\tRMSE = %.3f\tPSNR = %.3f\tTIME = %f (%f)\n", sqrtf(mse), mse_to_psnr(mse), estimate.avg, estimate.min);


    return true;
}

// Kodak image set from: http://r0k.us/graphics/kodak/
static const char * images[] = {
    "data/kodim01.png",
    "data/kodim02.png",
    "data/kodim03.png",
    "data/kodim04.png",
    "data/kodim05.png",
    "data/kodim06.png",
    "data/kodim07.png",
    "data/kodim08.png",
    "data/kodim09.png",
    "data/kodim10.png",
    "data/kodim11.png",
    "data/kodim12.png",
    "data/kodim13.png",
    "data/kodim14.png",
    "data/kodim15.png",
    "data/kodim16.png",
    "data/kodim17.png",
    "data/kodim18.png",
    "data/kodim19.png",
    "data/kodim20.png",
    "data/kodim21.png",
    "data/kodim22.png",
    "data/kodim23.png",
    "data/kodim24.png",
};
static const int image_count = sizeof(images) / sizeof(images[0]);


int main(int argc, char * argv[]) {

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-dds") == 0) {
            output_dds = true;
        }
        else if (strcmp(argv[i], "-ktx") == 0) {
            output_ktx = true;
        }
        else if (strcmp(argv[i], "-png") == 0) {
            output_png = true;
        }
        else if (strncmp(argv[i], "-q", 2) == 0) {
            if (argv[i][2]) {
                quality_level = (icbc::Quality)(argv[i][2] - '1');
                if (quality_level < icbc::Quality_Fast) quality_level = icbc::Quality_Fast;
                if (quality_level > icbc::Quality_Max) quality_level = icbc::Quality_Max;
            }
        }
        else if (atoi(argv[i])) {
            repeat_count = atoi(argv[i]);
        }
    }

    icbc::init_dxt1();
    int thread_count = ic::init_pfor();
    printf("Using %d threads.\n", thread_count);

    for (int i = 0; i < image_count; i++) {
        encode_image(images[i]);
    }

    total_mse /= total_block_count;

    printf("Average Results:\n");
    printf("\tRMSE = %.3f\tPSNR = %.3f\tTIME = %f (%f)\n", sqrtf(total_mse), mse_to_psnr(total_mse), total_avg_time, total_min_time);

    return 0;
}

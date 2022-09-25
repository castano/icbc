// ic_pfor v1.0 - Ignacio Castano <castano@gmail.com>
// LICENSE:
//  MIT License at the end of this file.

#ifndef IC_PFOR_H
#define IC_PFOR_H

// Allow disabling C++11 lambdas.
#ifndef IC_CC_LAMBDAS
#if defined(__GNUC__)
#define IC_CC_LAMBDAS __cplusplus>=201103L
#elif defined(__clang__)
#define IC_CC_LAMBDAS __has_feature(cxx_lambdas)
#else
#define IC_CC_LAMBDAS (_MSC_VER >= 1800)
#endif
#endif

namespace ic {

    // Init and destroy this library. Returns number of threads.
    int init_pfor(int worker_count = 0, bool use_calling_thread = true);
    void shut_pfor();

    // Invoke the given function pointer in parallel with idx values in the [0,count) range.
    typedef void ForTask(void * context, int idx);
    void pfor_run (ForTask * task, void * context, unsigned int count, unsigned int step = 1);

#if IC_CC_LAMBDAS
    // The lambda based body declaration is much nicer:
    // ic::pfor(count, step, [&](int i){ ... });
    template <typename F>
    void pfor(unsigned int count, unsigned int step, F f) {
        // Transform lambda into function pointer.
        auto lambda = [](void* context, int idx) {
            F & f = *reinterpret_cast<F *>(context);
            f(idx);
        };

        pfor_run(lambda, &f, count, step);
    }

    // Some shenanigas for a slightly better syntax:
    // ic_pfor(idx, count, step) { ... }
    template<typename F>
    struct PForRun {
        F f;
        unsigned int count;
        unsigned int step;
        PForRun(unsigned int count, unsigned int step, F f):f(f) {
            pfor(count, step, f);
        }
    private:
        PForRun& operator=(const PForRun&);
    };
    struct PForHelp {
        unsigned int count;
        unsigned int step;
        PForHelp(unsigned int count, unsigned int step) : count(count), step(step) {}
        template<typename F> PForRun<F> operator+(F f) { return PForRun<F>(count, step, f); }
    };

    //#define ic_pfor(IDX, COUNT) const auto& CONCAT(pfor__, __LINE__) = ic::PForHelp(COUNT, 1) + [&](int IDX)
    #define ic_pfor(IDX, COUNT, STEP) const auto& CONCAT(pfor__, __LINE__) = ic::PForHelp(COUNT, STEP) + [&](int IDX)

#endif // IC_CC_LAMBDAS

} // ic

#endif // IC_PFOR_H

#ifdef IC_PFOR_IMPLEMENTATION

// Maximum thread count is fixed, but can be tweaked with this definition:
#ifndef IC_MAX_THREAD_COUNT
#define IC_MAX_THREAD_COUNT 64
#endif

#ifndef IC_THREAD_STACK_SIZE
#define IC_THREAD_STACK_SIZE 0 // Use default size.
#endif

// Set this to 1 to use the Windows CRT safely inside the threads.
#ifndef IC_INIT_THREAD_CRT
#define IC_INIT_THREAD_CRT 0
#endif

#ifndef IC_ASSERT
#define IC_ASSERT assert
#include <assert.h>
#endif

#define IC_STATIC_ASSERT(x) static_assert(x, #x)

#if ((defined(_WIN32) || defined WIN32 || defined __NT__ || defined __WIN32__) && !defined __CYGWIN__)
#define IC_OS_WINDOWS 1
#endif
#if (defined linux || defined __linux__)
#define IC_OS_LINUX 1
#endif
#if defined(__NetBSD__)
#define IC_OS_NETBSD 1
#endif
#if defined(__APPLE__) || defined (__MACH__)
#define IC_OS_DARWIN 1
#endif
#if defined(__FreeBSD__)
#define IC_OS_FREEBSD 1
#endif
#if defined(__OpenBSD__)
#define IC_OS_OPENBSD 1
#endif
#if defined(__CYGWIN__)
#define IC_OS_CYGWIN 1
#endif
#if defined(__EMSCRIPTEN_PTHREADS__)
#define IC_OS_EMSCRIPTEN 1
#endif

#if IC_OS_WINDOWS || IC_OS_CYGWIN
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include <windows.h>
#if IC_INIT_THREAD_CRT
#include <process.h>
#endif
#endif

#if !IC_OS_WINDOWS
#include <pthread.h>
#include <unistd.h> // sysconf
#endif

#if IC_OS_DARWIN
#import <mach/mach_host.h>
#import <sys/sysctl.h>
#endif

#if IC_OS_EMSCRIPTEN
#include <emscripten/threading.h>
#endif

#include <stdint.h>
#include <stdio.h> // snprintf



#define IC_MAX_THREAD_NAME_LENGTH 32

namespace ic {

typedef uint32_t uint;
typedef uint32_t uint32;

/// Return the minimum of two values.
template <typename T>
//inline const T & min(const T & a, const T & b)
inline T min(const T & a, const T & b)
{
    return (a < b) ? a : b;
}



////////////////////////////////////////////////////////
// Atomics

#if _MSC_VER

#include <intrin.h>

#pragma intrinsic(__cpuid)
#pragma intrinsic(_WriteBarrier)
#define compiler_write_barrier      _WriteBarrier

#pragma intrinsic(_ReadWriteBarrier)
#define compiler_rw_barrier         _ReadWriteBarrier

#if _MSC_VER >= 1400  // ReadBarrier is VC2005
#pragma intrinsic(_ReadBarrier)
#define compiler_read_barrier       _ReadBarrier
#else
#define compiler_read_barrier       _ReadWriteBarrier
#endif

#else // GCC

#define compiler_rw_barrier()       asm volatile("" ::: "memory")
#define compiler_read_barrier       compiler_rw_barrier
#define compiler_write_barrier      compiler_rw_barrier

#endif

template <typename T>
inline void store_release_pointer(volatile T * pTo, T from) {
    IC_STATIC_ASSERT(sizeof(T) == sizeof(intptr_t));
    IC_ASSERT((((intptr_t)pTo) % sizeof(intptr_t)) == 0);
    IC_ASSERT((((intptr_t)&from) % sizeof(intptr_t)) == 0);

    compiler_write_barrier();
    *pTo = from;    // on x86, stores are Release
}

template <typename T>
inline T load_acquire_pointer(volatile T * ptr) {
    IC_STATIC_ASSERT(sizeof(T) == sizeof(intptr_t));
    IC_ASSERT((((intptr_t)ptr) % sizeof(intptr_t)) == 0);

    T ret = *ptr;   // on x86, loads are Acquire
    compiler_read_barrier();
    return ret;
}

#undef compiler_rw_barrier
#undef compiler_read_barrier
#undef compiler_write_barrier

#if IC_OS_WINDOWS

// Returns original value before addition.
inline uint32 atomic_fetch_and_add(uint32 * value, uint32 value_to_add) {
    IC_ASSERT((intptr_t(value) & 3) == 0);
    return uint32(_InterlockedExchangeAdd((long*)value, (long)value_to_add));
}

#else

// Returns original value before addition.
inline uint32 atomic_fetch_and_add(uint32 * value, uint32 value_to_add) {
    IC_ASSERT((intptr_t(value) & 3) == 0);
    return __sync_fetch_and_add(value, value_to_add);
}

#endif


////////////////////////////////////////////////////////
// System

#if IC_OS_WINDOWS || IC_OS_CYGWIN
typedef BOOL (WINAPI *LPFN_GSI)(LPSYSTEM_INFO);
typedef BOOL (WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);

static bool isWow64() {
    LPFN_ISWOW64PROCESS fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "IsWow64Process");

    BOOL wow64 = FALSE;

    if (NULL != fnIsWow64Process) {
        if (!fnIsWow64Process(GetCurrentProcess(), &wow64)) {
            // If error, assume false.
        }
    }

    return wow64 != 0;
}

static void getSystemInfo(SYSTEM_INFO * sysinfo) {
    BOOL success = FALSE;

    if (isWow64()) {
        LPFN_GSI fnGetNativeSystemInfo = (LPFN_GSI)GetProcAddress(GetModuleHandle(TEXT("kernel32")), "GetNativeSystemInfo");

        if (fnGetNativeSystemInfo != NULL) {
            success = fnGetNativeSystemInfo(sysinfo);
        }
    }

    if (!success) {
        GetSystemInfo(sysinfo);
    }
}
#endif

// Find the number of logical processors in the system.
// Based on: http://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
static int get_processor_count() {
#if IC_OS_WINDOWS || IC_OS_CYGWIN
    SYSTEM_INFO sysinfo;
    getSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;

    // Respect process affinity mask?
    DWORD_PTR pam, sam;
    GetProcessAffinityMask(GetCurrentProcess(), &pam, &sam);

    // Count number of bits set in the processor affinity mask.
    int count = 0;
    for (int i = 0; i < sizeof(DWORD_PTR) * 8; i++) {
        if (pam & (DWORD_PTR(1) << i)) count += 1;
    }
    IC_ASSERT((DWORD)count <= sysinfo.dwNumberOfProcessors);

    return count;
#elif IC_OS_LINUX || IC_OS_NETBSD // Linux, Solaris, & AIX
    return sysconf(_SC_NPROCESSORS_ONLN);
#elif IC_OS_DARWIN || IC_OS_FREEBSD || IC_OS_OPENBSD
    int numCPU;
    int mib[4];
    size_t len = sizeof(numCPU);

    // set the mib for hw.ncpu
    mib[0] = CTL_HW;

#if IC_OS_OPENBSD || IC_OS_FREEBSD
    mib[1] = HW_NCPU;
#else
    mib[1] = HW_AVAILCPU;
#endif

    // get the number of CPUs from the system
    sysctl(mib, 2, &numCPU, &len, NULL, 0);

    if (numCPU < 1) {
        mib[1] = HW_NCPU;
        sysctl( mib, 2, &numCPU, &len, NULL, 0 );

        if (numCPU < 1) {
            return 1; // Assume single core.
        }
    }

    return numCPU;
#elif IC_OS_EMSCRIPTEN
    if (!emscripten_has_threading_support()) return 1;
    return emscripten_num_logical_cores();
#else
    return 1; // Assume single core.
#endif
}


////////////////////////////////////////////////////////
// Thread

typedef void ThreadFunc(void * arg);

struct Thread {
#if IC_OS_WINDOWS
    HANDLE handle;
#else // POSIX
    pthread_t handle;
#endif

    char name[IC_MAX_THREAD_NAME_LENGTH];
    ThreadFunc * func;
    void * arg;
};


#if IC_OS_WINDOWS

// SetThreadName implementation from msdn:
// http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx

#pragma pack(push,8)
struct THREADNAME_INFO
{
    DWORD dwType; // Must be 0x1000.
    LPCSTR szName; // Pointer to name (in user addr space).
    DWORD dwThreadID; // Thread ID (-1=caller thread).
    DWORD dwFlags; // Reserved for future use, must be zero.
};
#pragma pack(pop)

static void setThreadName(DWORD dwThreadID, const char* threadName)
{
    static const DWORD MS_VC_EXCEPTION = 0x406D1388;

    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = threadName;
    info.dwThreadID = dwThreadID;
    info.dwFlags = 0;

    __try
    {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    }
    __except(EXCEPTION_EXECUTE_HANDLER)
    {
    }
}


#if IC_INIT_THREAD_CRT
static unsigned __cdecl threadFunc(void * arg)
#else
static unsigned long __stdcall threadFunc(void * arg)
#endif
{
    Thread * thread = (Thread *)arg;
    DWORD id = GetCurrentThreadId();
    setThreadName(id, thread->name);
    #ifdef IC_THREAD_NAME
    IC_THREAD_NAME(id, thread->name);
    #endif
    thread->func(thread->arg);
    return 0;
}

static void thread_start(Thread * thread, ThreadFunc * func, void * arg)
{
    thread->func = func;
    thread->arg = arg;
#if IC_INIT_THREAD_CRT
    thread->handle = (HANDLE)_beginthreadex(NULL, IC_THREAD_STACK_SIZE, threadFunc, thread, 0, NULL);
#else
    thread->handle = CreateThread(NULL, IC_THREAD_STACK_SIZE, threadFunc, thread, 0, NULL);
#endif
    IC_ASSERT(thread->handle != NULL);
}

static void thread_wait(Thread * thread)
{
    DWORD status = WaitForSingleObject (thread->handle, INFINITE);
    IC_ASSERT (status ==  WAIT_OBJECT_0);

    BOOL ok = CloseHandle (thread->handle);
    IC_ASSERT (ok);
    thread->handle = NULL;
}

#else // POSIX

static void * threadFunc(void * arg)
{
    Thread * thread = (Thread *)arg;
    thread->func(thread->arg);
    pthread_exit(0);
}

static void thread_start(Thread * thread, ThreadFunc * func, void * arg)
{
    thread->func = func;
    thread->arg = arg;
    int result = pthread_create(&thread->handle, NULL, threadFunc, thread);
    IC_ASSERT(result == 0);
}

static void thread_wait(Thread * thread)
{
    int result = pthread_join(thread->handle, NULL);
    thread->handle = 0;
    IC_ASSERT(result == 0);
}

#endif

static void thread_wait(Thread threads[], uint count)
{
    for (uint i = 0; i < count; i++) {
        thread_wait(&threads[i]);
    }
}


////////////////////////////////////////////////////////
// Event

#if IC_OS_WINDOWS

struct Event {
    HANDLE handle;
};

static void event_create(Event * event)
{
    event->handle = CreateEvent(/*lpEventAttributes=*/NULL, /*bManualReset=*/FALSE, /*bInitialState=*/FALSE, /*lpName=*/NULL);
}

static void event_destroy(Event * event)
{
    CloseHandle(event->handle);
    event->handle = NULL;
}

static void event_post(Event * event)
{
    SetEvent(event->handle);
}

static void event_wait(Event * event)
{
    WaitForSingleObject(event->handle, INFINITE);
}

#else // POSIX

struct Event {
    pthread_cond_t pt_cond;
    pthread_mutex_t pt_mutex;
    int count = 0;
    int wait_count = 0;
};

static void event_create(Event * event)
{
    event->count = 0;
    event->wait_count = 0;
    pthread_mutex_init(&event->pt_mutex, NULL);
    pthread_cond_init(&event->pt_cond, NULL);
}

static void event_destroy(Event * event)
{
    pthread_cond_destroy(&event->pt_cond);
    pthread_mutex_destroy(&event->pt_mutex);
}

static void event_post(Event * event)
{
    pthread_mutex_lock(&event->pt_mutex);

    event->count += 1;

    if (event->wait_count > 0) {
        pthread_cond_signal(&event->pt_cond);
    }

    pthread_mutex_unlock(&event->pt_mutex);
}

static void event_wait(Event * event)
{
    pthread_mutex_lock(&event->pt_mutex);

    while (event->count == 0) {
        event->wait_count += 1;
        pthread_cond_wait(&event->pt_cond, &event->pt_mutex);
        event->wait_count -= 1;
    }
    event->count -= 1;

    pthread_mutex_unlock(&event->pt_mutex);
}

#endif

static void event_post(Event threads[], uint count)
{
    for (uint i = 0; i < count; i++) {
        event_post(&threads[i]);
    }
}

static void event_wait(Event threads[], uint count)
{
    for (uint i = 0; i < count; i++) {
        event_wait(&threads[i]);
    }
}


////////////////////////////////////////////////////////
// Thread Pool

typedef void ThreadTask(void * context, int id);

struct ThreadPool {
    bool use_calling_thread;
    int worker_count;

    Thread workers[IC_MAX_THREAD_COUNT];
    Event startEvents[IC_MAX_THREAD_COUNT];
    Event finishEvents[IC_MAX_THREAD_COUNT];

    ThreadTask * func;
    void * arg;
};

static ThreadPool pool;

static void pool_func(void * arg) {
    uint i = uint((uintptr_t)arg); // This is OK, because workerCount should always be much smaller than 2^32

    while (true)
    {
        event_wait(&pool.startEvents[i]);

        ThreadTask * func = load_acquire_pointer(&pool.func);

        if (func == NULL) {
            return;
        }

        func(pool.arg, i + pool.use_calling_thread);

        event_post(&pool.finishEvents[i]);
    }
}

void thread_pool_run(ThreadTask * func, void * arg)
{
    // Set our desired function.
    store_release_pointer(&pool.func, func);
    store_release_pointer(&pool.arg, arg);

    // Resume threads.
    event_post(pool.startEvents, pool.worker_count - pool.use_calling_thread);

    if (pool.use_calling_thread) {
        func(arg, 0);
    }

    // Wait for threads to complete.
    event_wait(pool.finishEvents, pool.worker_count - pool.use_calling_thread);
}

int init_pfor(int worker_count, bool use_calling_thread) {

    if (worker_count <= 0) {
        worker_count = get_processor_count();
    }
    if (worker_count - use_calling_thread > IC_MAX_THREAD_COUNT) {
        worker_count = IC_MAX_THREAD_COUNT + use_calling_thread;
    }

    pool.worker_count = worker_count;
    pool.use_calling_thread = use_calling_thread;

    for (int i = 0; i < worker_count - use_calling_thread; i++) {
        event_create(&pool.startEvents[i]);
        event_create(&pool.finishEvents[i]);
    }

    for (int i = 0; i < worker_count - use_calling_thread; i++) {
        snprintf(pool.workers[i].name, IC_MAX_THREAD_NAME_LENGTH, "ic_pfor_worker %d", i);
        thread_start(&pool.workers[i], pool_func, (void*)(uintptr_t)(i));
    }

    return worker_count;
}

void shut_pfor() {

    // Set threads to terminate.
    store_release_pointer(&pool.func, (ThreadTask *)NULL);
    store_release_pointer(&pool.arg, (void *)NULL);

    // Resume threads.
    event_post(pool.startEvents, pool.worker_count - pool.use_calling_thread);

    // Wait until threads actually exit.
    thread_wait(pool.workers, pool.worker_count - pool.use_calling_thread);

    for (int i = 0; i < pool.worker_count - pool.use_calling_thread; i++) {
        event_destroy(&pool.startEvents[i]);
        event_destroy(&pool.finishEvents[i]);
    }
}


////////////////////////////////////////////////////////
// Parallel For

struct ParallelFor {
    ForTask * func;
    void * ctx;

    uint count;
    uint step;
    /*atomic*/ uint idx;
};

static ParallelFor pf;

static void pf_func(void * arg, int tid) {
    while (true) {
        uint new_idx = atomic_fetch_and_add(&pf.idx, pf.step);
        if (new_idx >= pf.count) {
            break;
        }

        const uint count = min(pf.count, new_idx + pf.step);
        for (uint i = new_idx; i < count; i++) {
            pf.func(pf.ctx, i);
        }
    }
}

void pfor_run(ForTask * task, void * context, uint count, uint step/*= 1*/) {

    pf.func = task;
    pf.ctx = context;

    // Init for loop state.
    pf.count = count;
    pf.step = step;
    pf.idx = 0;

    // Start pool threads.
    thread_pool_run(pf_func, NULL);

    IC_ASSERT(pf.idx >= pf.count);
}

} // ic
#endif // IC_PFOR_IMPLEMENTATION

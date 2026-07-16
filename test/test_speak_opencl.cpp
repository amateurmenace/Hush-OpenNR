// Speak GPU parity test (OpenCL) — runs the REAL RunOpenCLSpeak entry point
// against the CPU reference (speak_core.h). OpenCL is Resolve's render path on
// AMD/Intel/NVIDIA-without-CUDA and the primary Windows/Linux path, so this
// gives a THIRD verified backend (CPU + Metal + OpenCL); CUDA stays a faithful
// hardware-unverified port like Hush's.
//
// Build (macOS): c++ -O2 -std=c++14 -I../plugin test_speak_opencl.cpp \
//     ../plugin/SpeakOpenCLKernel.cpp -framework OpenCL -o test_speak_opencl

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <cmath>
#include <vector>

#include "SpeakParams.h"
#include "speak_core.h"

extern void RunOpenCLSpeak(void* p_CmdQ, int p_Width, int p_Height,
                           const SpeakParams& p_Params, const float* p_Src, float* p_Dst);

static int g_fail = 0;

static std::vector<float> makeFrame(int W, int H)
{
    std::vector<float> f(static_cast<size_t>(W) * H * 4);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const float u = static_cast<float>(x) / (W - 1);
            const float v = static_cast<float>(y) / (H - 1);
            f[i + 0] = 0.02f + 1.15f * u * u;
            f[i + 1] = 0.05f + 0.9f * v;
            f[i + 2] = 0.03f + 0.8f * (0.5f * u + 0.5f * (1.0f - v));
            f[i + 3] = 0.25f + 0.5f * u;
        }
    return f;
}

static SpeakParams baseParams()
{
    SpeakParams p = {};
    p.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    p.outputMode = SPEAK_OUT_WORKING;
    p.enableTone = 1; p.strength = 1.0f; p.viewMode = SPEAK_VIEW_RESULT;
    p.profile = speakcore::neutralProfile();
    return p;
}
static SpeakProfile stockProfile()
{
    SpeakProfile p = speakcore::neutralProfile();
    p.negGamma[0] = 0.66f; p.negGamma[2] = 0.58f; p.prnGamma[1] = 2.8f;
    p.negSpeed[2] = -1.35f; p.printerLights[0] = 2.5f; p.printerLights[2] = -1.5f;
    p.printerMaster = 0.8f;
    return p;
}

static cl_context g_ctx; static cl_command_queue g_q; static cl_device_id g_dev;

// Apple's DEPRECATED OpenCL runtime miscompiles global int32 atomics: a minimal
// 1000-work-item atomic_inc over 4 bins returns ~1.7e9 in every bin instead of
// 250, even though the device advertises cl_khr_global_int32_base_atomics (plain
// __global writes on the same buffer are fine). The scope's measurement pass
// bins the frame with atomics, so on such a device its result is meaningless.
//
// We probe for it and SKIP the stats-dependent cases loudly rather than silently
// passing them or reporting a false parity failure — the kernel source is
// correct and IS verified here on CPU + Metal, and OpenCL's real targets
// (NVIDIA/AMD/Intel on Windows/Linux) have working atomics. macOS production
// renders via Metal, so no shipping path depends on this.
// (This very likely also explains Hush's long-standing ~2e-3 Apple-OpenCL
// divergence — its noise estimator accumulates histograms with atomics too.)
static bool atomicsWork()
{
    static const char* K =
        "__kernel void probe(volatile __global uint* s){ atomic_inc(&s[get_global_id(0) % 4]); }\n";
    cl_int e;
    cl_program pr = clCreateProgramWithSource(g_ctx, 1, &K, NULL, &e);
    if (clBuildProgram(pr, 1, &g_dev, NULL, NULL, NULL) != CL_SUCCESS) return false;
    cl_kernel k = clCreateKernel(pr, "probe", &e);
    cl_mem b = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 16, NULL, &e);
    cl_uint z = 0, out[4] = { 0, 0, 0, 0 };
    clEnqueueFillBuffer(g_q, b, &z, sizeof(cl_uint), 0, 16, 0, NULL, NULL);
    clSetKernelArg(k, 0, sizeof(cl_mem), &b);
    size_t g = 1000;
    clEnqueueNDRangeKernel(g_q, k, 1, NULL, &g, NULL, 0, NULL, NULL);
    clFinish(g_q);
    clEnqueueReadBuffer(g_q, b, CL_TRUE, 0, 16, out, 0, NULL, NULL);
    clReleaseMemObject(b); clReleaseKernel(k); clReleaseProgram(pr);
    return (out[0] + out[1] + out[2] + out[3]) == 1000u;
}

static void run(int W, int H, const SpeakParams& p, const char* label, int mode)
{
    std::vector<float> src = makeFrame(W, H);
    const size_t n = static_cast<size_t>(W) * H * 4;
    const size_t bytes = n * sizeof(float);

    std::vector<float> cpu(n);
    speakcore::speakFrame(src.data(), W, H, p, cpu.data());

    cl_int err;
    cl_mem srcBuf = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, src.data(), &err);
    cl_mem dstBuf = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

    RunOpenCLSpeak((void*)g_q, W, H, p, (const float*)srcBuf, (float*)dstBuf);
    clFinish(g_q);

    std::vector<float> gpu(n);
    clEnqueueReadBuffer(g_q, dstBuf, CL_TRUE, 0, bytes, gpu.data(), 0, NULL, NULL);

    double maxd = 0.0, sumd = 0.0; size_t nOver = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(gpu[i]) - cpu[i]);
        if (d > maxd) maxd = d; sumd += d; if (d > 5e-3) nOver++;
    }
    const double meand = sumd / n;
    bool pass;
    if (mode == 1)      pass = (meand < 5e-5) && (nOver <= 400);                 // scope: hudOK
    else if (mode == 2) pass = (meand < 1e-5) && (nOver <= 64) && (maxd < 0.30); // bake gamut-edge boundary flips
    else                pass = (maxd < 5e-3) && (meand < 1e-4);                  // strict
    printf("  [%s] %-30s max %.2e  mean %.2e  over %zu\n", pass ? "PASS" : "FAIL", label, maxd, meand, nOver);
    if (!pass) g_fail++;
    clReleaseMemObject(srcBuf); clReleaseMemObject(dstBuf);
}

int main()
{
    printf("=== Speak OpenCL parity ===\n");
    cl_platform_id plat; if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS) { printf("no platform\n"); return 0; }
    cl_device_id dev;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS &&
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) { printf("no device\n"); return 0; }
    cl_int err;
    g_dev = dev;
    g_ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    g_q = clCreateCommandQueue(g_ctx, dev, 0, &err);
    const bool atomicsOK = atomicsWork();
    if (!atomicsOK) {
        char nm[128] = { 0 };
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(nm) - 1, nm, NULL);
        printf("  NOTE: this device's OpenCL global int32 atomics are BROKEN (%s).\n"
               "        The scope's measurement pass depends on them, so stats-dependent\n"
               "        cases are SKIPPED here. The kernel is verified on CPU + Metal;\n"
               "        OpenCL's real targets (Win/Linux NVIDIA/AMD/Intel) are unaffected.\n", nm);
    }
    const int W = 640, H = 480;

    { SpeakParams p = baseParams(); p.strength = 0.0f; run(W, H, p, "identity (strength 0)", 0); }
    { SpeakParams p = baseParams(); run(W, H, p, "tone neutral s1.0", 0); }
    { SpeakParams p = baseParams(); p.profile = stockProfile(); p.strength = 0.7f; run(W, H, p, "stock + printerLights s0.7", 0); }
    { SpeakParams p = baseParams(); p.inputColorSpace = SPEAK_CS_REC709_G24; p.profile = stockProfile(); run(W, H, p, "stock Rec709-in s1.0", 0); }
    { SpeakParams p = baseParams(); p.viewMode = SPEAK_VIEW_SPLIT; p.profile = stockProfile(); run(W, H, p, "split view s1.0", 0); }
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.profile = stockProfile(); run(W, H, p, "bake Rec.709 s1.0", 2); }
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.strength = 0.0f; run(W, H, p, "bake Rec.709 CST-only", 2); }
    { SpeakParams p = baseParams(); p.outputMode = SPEAK_OUT_BAKE_REC709; p.viewMode = SPEAK_VIEW_SPLIT; p.profile = stockProfile(); run(W, H, p, "bake + split view", 2); }
    { SpeakParams p = baseParams(); p.strength = 0.0f; p.enableDye = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.55f;
      speakcore::setDyeCoupler(p.profile, 1.0f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      run(W, H, p, "subtractive color standalone", 0); }
    { SpeakParams p = baseParams(); p.profile = stockProfile(); p.enableDye = 1;
      p.profile.subSat[0] = p.profile.subSat[1] = p.profile.subSat[2] = 0.8f;
      speakcore::setDyeCoupler(p.profile, 0.7f);
      p.profile.subSatKnee[0] = p.profile.subSatKnee[1] = p.profile.subSatKnee[2] = 2.2f;
      run(W, H, p, "tone + subtractive color", 0); }
    if (atomicsOK) {
        { SpeakParams p = baseParams(); p.scopeHD = 1; p.strength = 0.6f; p.profile = stockProfile(); run(W, H, p, "scope H&D on s0.6", 1); }
        { SpeakParams p = baseParams(); p.scopeDensity = 1; p.strength = 0.7f; p.profile = stockProfile(); run(W, H, p, "density scope on", 1); }
        { SpeakParams p = baseParams(); p.scopeHD = 1; p.scopeDensity = 1; p.strength = 0.7f; p.profile = stockProfile(); run(W, H, p, "both scopes on", 1); }
    } else {
        printf("  [SKIP] scope H&D / density / both    (device's OpenCL atomics are broken)\n");
    }

    printf("\n%s (%d failures)\n", g_fail ? "PARITY FAILED" : "PARITY GREEN", g_fail);
    return g_fail ? 1 : 0;
}

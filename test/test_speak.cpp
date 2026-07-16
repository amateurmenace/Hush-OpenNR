// Speak — CPU gate suite. Validates the density-spine math against the Phase-1
// control arms BEFORE the GPU ports:
//   G1 struct layout parity (the cardinal-rule layout check)
//   G2 color-management round-trip is lossless (the CST scaffold gate)
//   G3 identity at default (strength 0 => bit-exact pass-through)
//   G4 neutral-in => neutral-out is exact for a gray-balanced profile
//   G5 H&D curve + tone scale are monotone
//   G6 gray pivots to gray (18% in => 18% out) by construction
//   G7 the on-screen scope curve is sampled from the production kernel
//      (plot == pixels)
//
// Build:
//   c++ -O2 -std=c++14 -I../plugin test_speak.cpp -o test_speak && ./test_speak

#include <cstdio>
#include <cstddef>
#include <cmath>
#include <vector>
#include <cstdint>

#include "speak_core.h"

using namespace speakcore;

static int g_fail = 0;
// Gates that probe the pixel transform directly pass an empty stats block (the
// scope is off in those cases, so it is never read).
static uint32_t kNoStats[SPEAK_STATS_UINTS] = { 0 };
static void check(bool ok, const char* name, const char* detail = "")
{
    printf("  [%s] %s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) g_fail++;
}

// ----------------------------------------------------------------- G1 layout
static void gateLayout()
{
    printf("G1 struct layout parity\n");
    // All-4-byte-field invariant: sizeof must equal the field count * 4.
    const size_t profFields = 70;   // see SpeakParams.h; keep in sync (+3 halation)
    const size_t parFields  = 13 + profFields;
    check(sizeof(float) == 4 && sizeof(int) == 4, "float/int are 4 bytes");
    check(sizeof(SpeakProfile) == profFields * 4, "sizeof(SpeakProfile)==280",
          (std::to_string(sizeof(SpeakProfile))).c_str());
    check(sizeof(SpeakParams) == parFields * 4, "sizeof(SpeakParams)==332",
          (std::to_string(sizeof(SpeakParams))).c_str());
    check(offsetof(SpeakParams, profile) == 13 * 4, "profile offset==52",
          (std::to_string(offsetof(SpeakParams, profile))).c_str());
    // A few anchor offsets the GPU struct declarations must match.
    check(offsetof(SpeakProfile, printerLights) == 18 * 4, "printerLights offset==72");
    check(offsetof(SpeakProfile, prnDmin) == 22 * 4, "prnDmin offset==88");
    check(offsetof(SpeakProfile, dyeCouple) == 40 * 4, "dyeCouple offset==160");
}

// ------------------------------------------------------------ G2 round-trip
static void gateRoundTrip()
{
    printf("G2 color-management round-trip is lossless\n");
    const int spaces[] = { SPEAK_CS_DWG_INTERMEDIATE, SPEAK_CS_REC709_G24,
                           SPEAK_CS_ACESCCT, SPEAK_CS_LINEAR };
    for (int si = 0; si < 4; ++si) {
        const int cs = spaces[si];
        float maxErr = 0.0f;
        for (int i = 0; i <= 4000; ++i) {
            const float L = std::pow(10.0f, -4.0f + 8.0f * (i / 4000.0f)); // 1e-4..1e4
            const float v = encodeFromLinear(cs, L);
            const float L2 = decodeToLinear(cs, v);
            const float rel = std::fabs(L2 - L) / (std::fabs(L) + 1e-6f);
            if (rel > maxErr) maxErr = rel;
        }
        char buf[64]; snprintf(buf, sizeof(buf), "cs=%d maxRelErr=%.2e", cs, maxErr);
        check(maxErr < 1e-4f, "encode->decode round-trips", buf);
    }
    // Verify DI continuity at the segment cut (encode branches must meet).
    const float aLin = kDI_LIN_CUT * kDI_M;
    const float aLog = (std::log2(kDI_LIN_CUT + kDI_A) + kDI_B) * kDI_C;
    check(std::fabs(aLin - aLog) < 2e-4f, "DI encode is continuous at the cut");
}

// ----------------------------------------------------------- G3 identity
static void gateIdentity()
{
    printf("G3 identity at default (strength 0)\n");
    SpeakParams pr = {};
    pr.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    pr.enableTone = 1;
    pr.strength = 0.0f;               // default: no look
    pr.profile = neutralProfile();
    // A deterministic pseudo-random-ish tile of values.
    const int W = 17, H = 11;
    std::vector<float> src(W * H * 4), dst(W * H * 4);
    for (int i = 0; i < W * H * 4; ++i)
        src[i] = std::fmod(std::sin(i * 12.9898f) * 43758.5453f, 1.0f) * 0.5f + 0.5f;
    speakFrame(src.data(), W, H, pr, dst.data());
    float maxAbs = 0.0f;
    for (int i = 0; i < W * H * 4; ++i)
        maxAbs = std::fmax(maxAbs, std::fabs(dst[i] - src[i]));
    check(maxAbs == 0.0f, "strength 0 => bit-exact pass-through",
          (std::string("maxAbs=") + std::to_string(maxAbs)).c_str());

    // enableTone 0 with strength 1 is also identity.
    pr.strength = 1.0f; pr.enableTone = 0;
    speakFrame(src.data(), W, H, pr, dst.data());
    maxAbs = 0.0f;
    for (int i = 0; i < W * H * 4; ++i) maxAbs = std::fmax(maxAbs, std::fabs(dst[i] - src[i]));
    check(maxAbs == 0.0f, "enableTone 0 => bit-exact pass-through");
}

// ----------------------------------------------------------- G4 neutral
static void gateNeutral()
{
    printf("G4 neutral-in => neutral-out is exact (gray-balanced profile)\n");
    SpeakProfile p = neutralProfile();
    float maxChroma = 0.0f;
    for (int i = 0; i <= 500; ++i) {
        const float lin = std::pow(10.0f, -3.0f + 5.0f * (i / 500.0f)); // 1e-3..1e2
        const float oR = toneChannel(lin, 0, p);
        const float oG = toneChannel(lin, 1, p);
        const float oB = toneChannel(lin, 2, p);
        maxChroma = std::fmax(maxChroma, std::fmax(std::fabs(oR - oG), std::fabs(oG - oB)));
    }
    check(maxChroma < 1e-6f, "R==G==B out for R==G==B in",
          (std::string("maxChroma=") + std::to_string(maxChroma)).c_str());
}

// ----------------------------------------------------------- G5 monotone
static void gateMonotone()
{
    printf("G5 H&D curve + tone scale are monotone\n");
    SpeakProfile p = neutralProfile();
    // H&D curve monotone in logH.
    float prev = -1e30f; bool mono = true;
    for (int i = 0; i <= 6000; ++i) {
        const float logH = -6.0f + 12.0f * (i / 6000.0f);
        const float D = hdCurve(logH, p.negDmin[0], p.negDmax[0], p.negGamma[0],
                                p.negToe[0], p.negShoulder[0], p.negSpeed[0]);
        if (D < prev - 1e-6f) { mono = false; break; }
        prev = D;
    }
    check(mono, "hdCurve is non-decreasing in logH");

    // Full tone scale monotone in scene-linear.
    prev = -1e30f; mono = true;
    for (int i = 0; i <= 6000; ++i) {
        const float lin = std::pow(10.0f, -4.0f + 8.0f * (i / 6000.0f));
        const float o = toneChannel(lin, 0, p);
        if (o < prev - 1e-7f) { mono = false; break; }
        prev = o;
    }
    check(mono, "toneChannel is non-decreasing in scene-linear");

    // The curve is a real S (has contrast), not a straight identity. Measure
    // the mid-gray slope over a tight +/-0.25 stop window (the design contrast;
    // a wide window dips into toe/shoulder and reads lower, as real film does).
    const float dS = 0.25f;
    const float sysGamma = (std::log2(toneChannel(k18Gray * std::exp2(dS), 0, p) / k18Gray) -
                            std::log2(toneChannel(k18Gray * std::exp2(-dS), 0, p) / k18Gray)) / (2.0f * dS);
    char buf[48]; snprintf(buf, sizeof(buf), "systemGamma~%.2f", sysGamma);
    check(sysGamma > 1.15f && sysGamma < 2.4f, "system gamma is filmic (~1.6)", buf);
}

// ----------------------------------------------------------- G6 gray pivot
static void gateGrayPivot()
{
    printf("G6 gray pivots to gray (18%% in => 18%% out)\n");
    SpeakProfile p = neutralProfile();
    // Also test an intentionally un-balanced (per-channel) profile: the pivot
    // is per-channel, so gray still maps to 0.18 on every channel.
    p.negGamma[0] = 0.70f; p.prnGamma[2] = 2.9f; p.negSpeed[1] = -2.2f;
    float maxErr = 0.0f;
    for (int ch = 0; ch < 3; ++ch)
        maxErr = std::fmax(maxErr, std::fabs(toneChannel(k18Gray, ch, p) - k18Gray));
    check(maxErr < 1e-5f, "each channel maps 0.18 -> 0.18",
          (std::string("maxErr=") + std::to_string(maxErr)).c_str());
}

// ----------------------------------------------------------- G7 scope==kernel
static void gateScopeMatchesKernel()
{
    printf("G7 scope curve tracks the pixels at every strength (plot == pixels)\n");
    SpeakProfile prof = neutralProfile();
    prof.negGamma[0] = 0.66f; prof.prnGamma[1] = 2.7f;  // non-trivial, per-channel
    const float strengths[] = { 0.0f, 0.5f, 1.0f };     // incl. identity (s=0)
    float maxErr = 0.0f;
    for (int si = 0; si < 3; ++si) {
        SpeakParams pr = {};
        pr.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
        pr.enableTone = 1; pr.strength = strengths[si]; pr.viewMode = SPEAK_VIEW_RESULT;
        pr.scopeHD = 0;                                  // measure the transform, not the overlay
        pr.profile = prof;
        for (int i = 0; i <= 200; ++i) {
            const float inStops = -6.0f + 12.0f * (i / 200.0f);
            const float lin = k18Gray * std::exp2(inStops);
            const float enc = diEncode(lin);
            float oR, oG, oB;
            processPixel(enc, enc, enc, 0.0f, 0.0f, 0.0f, 4, 4, 100, 100, pr, kNoStats,oR, oG, oB); // the REAL pixel path
            for (int ch = 0; ch < 3; ++ch) {
                const float scopeOut = scopeYStops(inStops, ch, pr);
                const float outCh = (ch == 0) ? oR : (ch == 1) ? oG : oB;
                const float pxLin = decodeToLinear(pr.inputColorSpace, outCh);
                const float pxStops = std::log2((pxLin < kLinTiny ? kLinTiny : pxLin) / k18Gray);
                maxErr = std::fmax(maxErr, std::fabs(scopeOut - pxStops));
            }
        }
    }
    // Bounded by the CST encode/decode round-trip (~1e-6), not a scope discrepancy.
    check(maxErr < 1e-4f, "scope value == pixel value at every strength",
          (std::string("maxErr=") + std::to_string(maxErr)).c_str());
}

// -------------------------------------------------------------- G8 bake CST
static void gateBakeCST()
{
    printf("G8 Bake-to-Rec.709 CST scaffold (neutral-identity + round-trip)\n");
    SpeakParams pr = {};
    pr.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    pr.outputMode = SPEAK_OUT_BAKE_REC709;
    pr.enableTone = 0; pr.strength = 0.0f;      // pure CST, no look
    pr.profile = neutralProfile();

    // Neutral in -> neutral out: a DWG gray ramp bakes to Rec.709 with equal
    // channels (bounded by the published matrix's own rounding).
    float maxChroma = 0.0f;
    for (int i = 0; i <= 400; ++i) {
        const float lin = std::pow(10.0f, -3.0f + 5.0f * (i / 400.0f));
        const float enc = diEncode(lin);
        float oR, oG, oB;
        processPixel(enc, enc, enc, 0.0f, 0.0f, 0.0f, 4, 4, 100, 100, pr, kNoStats,oR, oG, oB);
        maxChroma = std::fmax(maxChroma, std::fmax(std::fabs(oR - oG), std::fabs(oG - oB)));
    }
    check(maxChroma < 2e-3f, "DWG neutral bakes to Rec.709 neutral",
          (std::string("maxChroma=") + std::to_string(maxChroma)).c_str());

    // 18% gray bakes to the correct Rec.709 code value (pow(0.18, 1/2.4)).
    {
        const float enc = diEncode(k18Gray);
        float oR, oG, oB;
        processPixel(enc, enc, enc, 0.0f, 0.0f, 0.0f, 4, 4, 100, 100, pr, kNoStats,oR, oG, oB);
        const float expect = std::pow(k18Gray, 1.0f / 2.4f);
        check(std::fabs(oR - expect) < 3e-3f, "18% gray -> correct Rec.709 code",
              (std::string("got=") + std::to_string(oR) + " want=" + std::to_string(expect)).c_str());
    }

    // Round-trip DWG-linear -> Rec.709-linear -> DWG-linear ~ identity (proves
    // the forward matrices are internally consistent).
    const float XYZ_to_DWG[9] = {
        1.51667205f,-0.28147806f,-0.14696364f,
       -0.46491710f, 1.25142377f, 0.17488461f,
        0.06484904f, 0.10913935f, 0.76141462f };
    const float Rec709_to_XYZ[9] = {
        0.41245643f, 0.35757608f, 0.18043748f,
        0.21267285f, 0.71515217f, 0.07217500f,
        0.01933390f, 0.11919203f, 0.95030407f };
    const float cols[4][3] = { {0.2f,0.2f,0.2f}, {0.4f,0.1f,0.05f}, {0.05f,0.3f,0.2f}, {0.6f,0.5f,0.1f} };
    float maxErr = 0.0f;
    for (int t = 0; t < 4; ++t) {
        float rr, rg, rb;
        gamutToRec709Lin(SPEAK_CS_DWG_LINEAR, cols[t][0], cols[t][1], cols[t][2], rr, rg, rb);
        float X, Y, Z, br, bg, bb;
        mul3(Rec709_to_XYZ, rr, rg, rb, X, Y, Z);
        mul3(XYZ_to_DWG, X, Y, Z, br, bg, bb);
        maxErr = std::fmax(maxErr, std::fmax(std::fabs(br - cols[t][0]),
                           std::fmax(std::fabs(bg - cols[t][1]), std::fabs(bb - cols[t][2]))));
    }
    check(maxErr < 1e-4f, "DWG->Rec.709->DWG round-trips",
          (std::string("maxErr=") + std::to_string(maxErr)).c_str());
}

// ------------------------------------------------------ G9 view delivery CST
static void gateViewDelivery()
{
    printf("G9 view overrides deliver through the output CST\n");
    const float encGray = diEncode(k18Gray);              // DI 18% gray ~= 0.336
    const float rec709Gray = std::pow(k18Gray, 1.0f / 2.4f); // Rec.709 ~= 0.489

    // Bake + Input view: shows the input DELIVERED to Rec.709 (no look) — a
    // valid "before" in the same space as the result, NOT the raw DI buffer
    // (that would put the two Split halves in different color spaces).
    SpeakParams pr = {};
    pr.inputColorSpace = SPEAK_CS_DWG_INTERMEDIATE;
    pr.outputMode = SPEAK_OUT_BAKE_REC709;
    pr.enableTone = 1; pr.strength = 1.0f;                // look on, but Input shows input w/o look
    pr.viewMode = SPEAK_VIEW_INPUT;
    pr.profile = neutralProfile();
    float oR, oG, oB;
    processPixel(encGray, encGray, encGray, 0.0f, 0.0f, 0.0f, 4, 4, 100, 100, pr, kNoStats,oR, oG, oB);
    check(std::fabs(oR - rec709Gray) < 3e-3f, "bake+Input shows input in Rec.709",
          (std::string("got=") + std::to_string(oR) + " want=" + std::to_string(rec709Gray)).c_str());
    check(std::fabs(oR - encGray) > 0.1f, "bake+Input is NOT the raw DI buffer");

    // Working + Input view: bit-exact raw input pass-through.
    pr.outputMode = SPEAK_OUT_WORKING;
    processPixel(encGray, encGray, encGray, 0.0f, 0.0f, 0.0f, 4, 4, 100, 100, pr, kNoStats,oR, oG, oB);
    check(oR == encGray, "working+Input is bit-exact raw input");

    // Bake + Split: left half (input) and right half (result) share Rec.709 —
    // the left-half pixel equals the delivered input, the right-half is baked.
    pr.outputMode = SPEAK_OUT_BAKE_REC709; pr.viewMode = SPEAK_VIEW_SPLIT;
    float lR, lG, lB, rR, rG, rB;
    processPixel(encGray, encGray, encGray, 0.0f, 0.0f, 0.0f, 10, 4, 100, 100, pr, kNoStats,lR, lG, lB);  // x<W/2 -> input
    processPixel(encGray, encGray, encGray, 0.0f, 0.0f, 0.0f, 90, 4, 100, 100, pr, kNoStats,rR, rG, rB);  // x>=W/2 -> result
    check(std::fabs(lR - rec709Gray) < 3e-3f, "bake+Split left half is delivered input (Rec.709)");
    check(std::fabs(rR - rec709Gray) < 6e-3f, "bake+Split right half is result (Rec.709, same space)");
}

// ------------------------------------------------- G12..G16 halation / pyramid
//
// These are BEHAVIOURAL gates, not parity gates. The scatter pyramid is the kind
// of module where every backend can agree on the same wrong answer (lesson L3):
// a wrong sigma ladder, a leaked normalisation, a threshold applied after the
// downsample, or a scatter-blind density scope would all keep parity green at
// ~2e-5 while shipping a wrong halo. Each gate below is written so that it FAILS
// on a specific plausible defect, and several assert the failure explicitly.

// Render a frame's scatter field through the real production builder.
static void buildScatterFor(const std::vector<float>& src, int W, int H,
                            const SpeakParams& pr, std::vector<float>& scat)
{
    std::vector<float> arena(static_cast<size_t>(halArenaPixels(W, H)) * 3, 0.0f);
    scat.assign(static_cast<size_t>(W) * H * 3, 0.0f);
    buildHalScatter(src.data(), W, H, pr, arena.data(), scat.data());
}
// A linear-space frame with a single bright impulse at the centre.
static std::vector<float> impulseFrame(int W, int H, float v)
{
    std::vector<float> f(static_cast<size_t>(W) * H * 4, 0.0f);
    const size_t i = ((static_cast<size_t>(H / 2)) * W + (W / 2)) * 4;
    f[i + 0] = v; f[i + 1] = v; f[i + 2] = v; f[i + 3] = 1.0f;
    return f;
}
static SpeakParams halParams(int cs, float amount, float radius, float thresh)
{
    SpeakParams pr = {};
    pr.inputColorSpace = cs;
    pr.outputMode = SPEAK_OUT_WORKING;
    pr.strength = 1.0f;
    pr.viewMode = SPEAK_VIEW_RESULT;
    pr.enableTone = 1; pr.enableDye = 1; pr.enableSplit = 1; pr.enableOptics = 1;
    pr.profile = neutralProfile();
    pr.profile.halAmount = amount;
    pr.profile.halRadius = radius;
    pr.profile.halThresh = thresh;
    return pr;
}

static void gateHalIdentity()
{
    printf("G12 halation identity + the enable gates\n");
    const int W = 64, H = 64;
    std::vector<float> src(static_cast<size_t>(W) * H * 4);
    for (size_t i = 0; i < src.size(); i += 4) {
        src[i + 0] = 0.3f + 0.5f * ((i / 4) % 7) / 7.0f;
        src[i + 1] = 0.9f; src[i + 2] = 0.2f; src[i + 3] = 1.0f;
    }
    std::vector<float> a(src.size()), b(src.size());

    // amount 0 must be BIT-EXACT with the pre-halation path.
    SpeakParams p0 = halParams(SPEAK_CS_LINEAR, 0.0f, 1.0f, 0.6f);
    SpeakParams p1 = halParams(SPEAK_CS_LINEAR, 0.8f, 1.0f, 0.6f);
    speakFrame(src.data(), W, H, p0, a.data());
    speakFrame(src.data(), W, H, p1, b.data());
    float maxD = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) maxD = std::fmax(maxD, std::fabs(a[i] - b[i]));
    check(maxD > 1e-4f, "G12a halation at amount 0.8 actually CHANGES the frame (not a no-op)",
          (std::string("maxDelta=") + std::to_string(maxD)).c_str());

    // enableOptics off must be bit-exact with amount 0 — the toggle must be real.
    SpeakParams p2 = p1; p2.enableOptics = 0;
    speakFrame(src.data(), W, H, p2, b.data());
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) mx = std::fmax(mx, std::fabs(a[i] - b[i]));
    check(mx == 0.0f, "G12b enableOptics=0 is BIT-EXACT with amount 0",
          (std::string("maxAbs=") + std::to_string(mx)).c_str());

    // strength 0 must stay bit-exact identity WITH halation cranked. This is the
    // trap: injecting into the dry side of the lerp would leave raw scatter added
    // in linear with no curve downstream — the end-chain overlay the arm rejected.
    SpeakParams p3 = halParams(SPEAK_CS_LINEAR, 1.0f, 2.0f, 0.6f);
    p3.strength = 0.0f; p3.enableDye = 0; p3.enableSplit = 0;
    speakFrame(src.data(), W, H, p3, b.data());
    float mi = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) mi = std::fmax(mi, std::fabs(src[i] - b[i]));
    check(mi == 0.0f, "G12c strength 0 stays BIT-EXACT identity with halation at full",
          (std::string("maxAbs=") + std::to_string(mi)).c_str());

    // halRadius 0 must not produce NaN (the sigma floor).
    SpeakParams p4 = halParams(SPEAK_CS_LINEAR, 1.0f, 0.0f, 0.6f);
    speakFrame(src.data(), W, H, p4, b.data());
    bool finite = true;
    for (size_t i = 0; i < b.size(); ++i) if (!std::isfinite(b[i])) finite = false;
    check(finite, "G12d halRadius=0 renders finite (the sigma floor holds)");
}

static void gateHalLadder()
{
    printf("G13 the pyramid's sigma ladder is what the core says it is\n");
    // A ladder off by one octave halves the shipped halo and is otherwise
    // invisible — nothing else in the suite would notice, and parity never
    // would. Pin halLevelSigma() against the MEASURED impulse response of each
    // level of the real production pyramid.
    //
    // Measure ONE LEVEL AT A TIME, not the mixture. The first version of this
    // gate took the second moment of the full mixture and failed on correct
    // code: an r^-3 skirt makes <r^2> diverge (in 2D the integrand r^2 * r^-3 * r
    // is flat in r), so the mixture's second moment is dominated by its widest
    // level and estimates nothing about the core. A single level IS near-
    // Gaussian, so <r^2> = 2*sigma^2 is a valid estimator there.
    const int W = 512, H = 512;
    std::vector<float> src = impulseFrame(W, H, 4000.0f);
    std::vector<float> arena(static_cast<size_t>(halArenaPixels(W, H)) * 3, 0.0f);
    std::vector<float> scat(static_cast<size_t>(W) * H * 3, 0.0f);
    SpeakParams pr = halParams(SPEAK_CS_LINEAR, 1.0f, 1.0f, 0.6f);
    buildHalScatter(src.data(), W, H, pr, arena.data(), scat.data());   // fills the arena

    float worst = 0.0f; int worstL = 0;
    for (int L = 1; L <= 5; ++L) {
        int lw, lh, off;
        halLevelInfo(W, H, L, lw, lh, off);
        double m0 = 0.0, m2 = 0.0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const double v = halSampleLevel(arena.data(), off, lw, lh, W, H, x, y, 0);
                const double dx = x - W / 2, dy = y - H / 2;
                m0 += v; m2 += v * (dx * dx + dy * dy);
            }
        const double sigMeas = std::sqrt(m2 / (m0 * 2.0));   // <r^2> = 2 sigma^2
        const double sigL = halLevelSigma(L);
        const double ratio = sigMeas / sigL;
        printf("    level %d: core says sigma=%6.2f   measured=%7.2f   ratio=%.3f\n",
               L, sigL, sigMeas, ratio);
        if (std::fabs(std::log2(ratio)) > worst) { worst = std::fabs(std::log2(ratio)); worstL = L; }
    }
    check(worst < 0.35f, "G13 each level's measured sigma matches halLevelSigma()",
          (std::string("worst |log2(ratio)|=") + std::to_string(worst) +
           " at level " + std::to_string(worstL)).c_str());
}

static void gateHalEnergy()
{
    printf("G14 the scatter pyramid conserves energy (interior)\n");
    // The mixture is a convex combination (sum w = 1) of mean-preserving levels,
    // so total scattered energy must equal total excess. Measured in the INTERIOR
    // only: clamp-to-edge loses energy at the border with no closed form to
    // compensate, so a whole-frame gate would fail on correct code.
    const int W = 512, H = 512;
    std::vector<float> src = impulseFrame(W, H, 100.0f), scat;
    SpeakParams pr = halParams(SPEAK_CS_LINEAR, 1.0f, 1.0f, 0.6f);
    buildScatterFor(src, W, H, pr, scat);
    const float excess = halExcess(100.0f, 0.6f);
    double sum = 0.0;
    for (size_t k = 0; k < static_cast<size_t>(W) * H; ++k) sum += scat[k * 3];
    const double rel = std::fabs(sum - excess) / excess;
    printf("    sum(scatter)=%.4f  excess=%.4f  rel=%.2e\n", sum, excess, rel);
    check(rel < 5e-3, "G14a scattered energy == source excess (impulse, interior)",
          (std::string("rel=") + std::to_string(rel)).c_str());

    // Sensitivity: an UNNORMALISED mixture would fail. Assert the normalisation
    // is load-bearing by checking the weights actually sum to something != 1.
    float wsum = 0.0f;
    const int nLev = halLevelCount(W, H);
    for (int L = 0; L < nLev; ++L) wsum += halLevelWeight(L, halSigmaPx(H, pr));
    check(wsum > 1.5f, "G14b the raw level weights do NOT sum to 1 (so G14a is a real gate)",
          (std::string("raw wsum=") + std::to_string(wsum)).c_str());
}

static void gateHalTail()
{
    printf("G15 multi-scale scatter has a wider skirt than a matched Gaussian\n");
    // THE CLAIM: an octave mixture gives a bright core PLUS a wide faint skirt,
    // which a single Gaussian cannot do at any sigma. Measure it — do not cite
    // glare literature for it (that models the eye, not an emulsion).
    const int W = 1024, H = 1024;
    std::vector<float> src = impulseFrame(W, H, 4000.0f), scat;
    SpeakParams pr = halParams(SPEAK_CS_LINEAR, 1.0f, 0.5f, 0.6f);
    buildScatterFor(src, W, H, pr, scat);

    // Radial profile of the mixture.
    auto radial = [&](float r0, float r1) {
        double s = 0.0; int n = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float dx = x - W / 2.0f, dy = y - H / 2.0f;
                const float d = std::sqrt(dx * dx + dy * dy);
                if (d >= r0 && d < r1) { s += scat[(static_cast<size_t>(y) * W + x) * 3]; n++; }
            }
        return n ? s / n : 0.0;
    };
    // Match a Gaussian to the mixture's CORE, then compare far-field.
    const double core = radial(0.0f, 2.0f);
    const double near = radial(4.0f, 6.0f);
    const double far  = radial(48.0f, 64.0f);
    // Fit sigma from the core:near ratio of a Gaussian, then predict its far field.
    const double r1 = 5.0, r2 = 56.0;
    const double sig2 = (r1 * r1) / (2.0 * std::log(core / (near > 0 ? near : 1e-30)));
    const double gaussFar = core * std::exp(-(r2 * r2) / (2.0 * sig2));
    printf("    core=%.3e  near(5px)=%.3e  far(56px)=%.3e | matched Gaussian far=%.3e\n",
           core, near, far, gaussFar);
    check(far > gaussFar * 100.0,
          "G15 the mixture's far field is >>100x a core-matched Gaussian's (a real skirt)",
          (std::string("ratio=") + std::to_string(far / (gaussFar > 0 ? gaussFar : 1e-300))).c_str());
}

static void gateHalResolution()
{
    printf("G16 the halo is resolution-independent (proxy == full res)\n");
    // Speak's radius is a % of frame HEIGHT precisely so a colourist can grade on
    // a proxy and deliver at full res. If the level bracket quantised the radius,
    // the halo would JUMP between resolutions — measure it instead of asserting.
    auto haloProfile = [](int W, int H) {
        std::vector<float> src(static_cast<size_t>(W) * H * 4, 0.0f);
        const int rad = H / 10;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float dx = x - W / 2.0f, dy = y - H / 2.0f;
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                const float v = (std::sqrt(dx * dx + dy * dy) <= rad) ? 40.0f : 0.0f;
                src[i] = v; src[i + 1] = v; src[i + 2] = v; src[i + 3] = 1.0f;
            }
        SpeakParams pr = halParams(SPEAK_CS_LINEAR, 1.0f, 2.0f, 0.6f);
        std::vector<float> scat;
        std::vector<float> arena(static_cast<size_t>(halArenaPixels(W, H)) * 3, 0.0f);
        scat.assign(static_cast<size_t>(W) * H * 3, 0.0f);
        buildHalScatter(src.data(), W, H, pr, arena.data(), scat.data());
        // Sample the scatter at fixed FRACTIONS of frame height along +x, with
        // BILINEAR interpolation. Integer sampling made this gate measure its
        // own rounding: at H=270 the k=5 probe truncated 33.75 -> 33, landing
        // 0.75 px nearer the disc, and on the halo's steepest gradient that
        // alone read as a 12% "resolution dependence" in correct code.
        std::vector<double> prof;
        for (int k = 1; k <= 6; ++k) {
            const double fx = W / 2.0 + (static_cast<double>(H) * k) / 40.0;
            const int py = H / 2;
            const int x0 = static_cast<int>(std::floor(fx));
            const double tx = fx - x0;
            const double a = scat[(static_cast<size_t>(py) * W + x0) * 3];
            const double b = scat[(static_cast<size_t>(py) * W + x0 + 1) * 3];
            prof.push_back(a * (1.0 - tx) + b * tx);
        }
        return prof;
    };
    std::vector<double> lo = haloProfile(480, 270);     // proxy
    std::vector<double> hi = haloProfile(1920, 1080);   // full res
    double worst = 0.0;
    for (size_t k = 0; k < lo.size(); ++k) {
        const double rel = std::fabs(lo[k] - hi[k]) / (std::fabs(hi[k]) + 1e-9);
        printf("    r=%.3fH  proxy=%.4f  full=%.4f  rel=%.3f\n",
               (k + 1) / 40.0, lo[k], hi[k], rel);
        if (rel > worst) worst = rel;
    }
    check(worst < 0.06, "G16 the halo profile matches across a 4x resolution change",
          (std::string("worst rel=") + std::to_string(worst)).c_str());
}

static void gateHalScopeSeesScatter()
{
    printf("G17 the density scope measures the HALATED result (the L3 trap)\n");
    // If the scope's measurement pass were scatter-blind, all four backends would
    // agree on the same wrong parade and parity would stay green at 2e-5. The
    // failure is directional: halation raises exposure, so density FALLS — a
    // blind scope would draw the halo darker than the pixels are.
    const int W = 128, H = 128;
    std::vector<float> src(static_cast<size_t>(W) * H * 4, 0.0f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            const bool hot = (x > 40 && x < 88 && y > 40 && y < 88);
            const float v = hot ? 60.0f : 0.02f;
            src[i] = v; src[i + 1] = v * 0.9f; src[i + 2] = v * 0.8f; src[i + 3] = 1.0f;
        }
    SpeakParams pr = halParams(SPEAK_CS_LINEAR, 1.0f, 3.0f, 0.6f);
    pr.scopeDensity = 1; pr.enableDye = 0; pr.enableSplit = 0;

    std::vector<float> arena(static_cast<size_t>(halArenaPixels(W, H)) * 3, 0.0f);
    std::vector<float> scat(static_cast<size_t>(W) * H * 3, 0.0f);
    buildHalScatter(src.data(), W, H, pr, arena.data(), scat.data());

    std::vector<uint32_t> withScat(SPEAK_STATS_UINTS), blind(SPEAK_STATS_UINTS);
    computeStats(src.data(), scat.data(), W, H, pr, withScat.data());
    computeStats(src.data(), nullptr,     W, H, pr, blind.data());   // the bug, simulated
    int diff = 0;
    for (int k = 0; k < SPEAK_WF_COLS * SPEAK_WF_ROWS * 3; ++k)
        if (withScat[SPEAK_STATS_WF + k] != blind[SPEAK_STATS_WF + k]) diff++;
    printf("    parade cells that differ when the scope is scatter-blind: %d\n", diff);
    check(diff > 0, "G17 a scatter-blind density parade is measurably WRONG (so passing scatter matters)",
          (std::string("cells=") + std::to_string(diff)).c_str());
}

int main()
{
    printf("=== Speak CPU gate suite ===\n");
    gateLayout();
    gateRoundTrip();
    gateIdentity();
    gateNeutral();
    gateMonotone();
    gateGrayPivot();
    gateScopeMatchesKernel();
    gateBakeCST();
    gateViewDelivery();
    gateHalIdentity();
    gateHalLadder();
    gateHalEnergy();
    gateHalTail();
    gateHalResolution();
    gateHalScopeSeesScatter();
    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL GATES GREEN", g_fail);
    return g_fail ? 1 : 0;
}

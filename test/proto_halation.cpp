// Speak — Phase 4 PROTOTYPE + control arm: red-weighted halation.
//
// Gate-first (the X3 law): this is a CPU prototype, NOT shipping code. It only
// earns a place in speak_core.h + the three kernels if it beats the cheap
// baseline on the stated measurement.
//
// THE PHYSICS CLAIM. Real halation is light that passes through the emulsion,
// reflects off the base, and RE-EXPOSES the negative from behind. So the scatter
// must be injected as EXPOSURE (additive in linear light) BEFORE the H&D curve —
// after which the curve's shoulder compresses it. It is red-weighted because the
// light that reaches the base has already been depleted of blue and green (by
// the yellow filter layer and the upper emulsion layers), and the red-sensitive
// layer sits closest to the base.
//
// THE CHEAP BASELINE: the thing every "film look" plugin actually ships — the
// same blurred highlight excess, red-tinted, added to the OUTPUT at the end of
// the chain. It bypasses the curve entirely.
//
// ---------------------------------------------------------------------------
// WHAT THIS ARM MEASURES, AND WHAT IT DELIBERATELY DOES NOT
//
// H1/H2 gate the INJECTION POINT (re-exposure vs end-chain overlay). They are
// the arm's original content and they hold.
//
// H3/H4 were added after an adversarial review found that H1/H2 gate the
// injection point AND NOTHING ELSE. Measured on the old arm: deleting the red
// weighting entirely (w = {1,1,1}) still passed 5/5; INVERTING it to blue
// (w = {0.10,0.30,1.0}) — the exact opposite of the module's headline physics —
// also passed 5/5. The old scene generator only ever built a near-white disc, so
// no chromatic claim was ever exercised. H3/H4 close that hole, and they are
// written so that they FAIL on the sabotaged models (asserted below, in-arm).
//
// H1a IS NOT A GATE. "The physical halo is bounded by paper white" cannot fail:
// hdCurve is bounded above by construction (d1 >= Dmin since softplus >= 0, and
// hdCurve is monotone in d1), so sup toneChannel is analytic and independent of
// every halation parameter. Measured: toneChannel(FLT_MAX) = 10.672234 = the
// closed form 0.18*10^(Dref - Dprn(-negDmax + printerOff)) exactly. It passed
// even at amount=0 and amount=50. It is reported here as a PROPERTY (with its
// closed form checked), not credited as evidence the halation model is right.
// The informative half is H1b: the baseline has no ceiling at all.
//
// Build: c++ -O2 -std=c++14 -I../plugin proto_halation.cpp -o proto_halation

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "speak_core.h"
using namespace speakcore;

static int g_fail = 0;
static void check(bool ok, const char* name, const std::string& detail = "")
{
    printf("  [%s] %s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) g_fail++;
}

// ---- separable blur (prototype; the shipping version becomes the pyramid) ----
static void blurSep(const std::vector<float>& in, int W, int H, float sigma, std::vector<float>& out)
{
    const int R = static_cast<int>(sigma * 3.0f) + 1;
    std::vector<float> k(2 * R + 1);
    float ksum = 0.0f;
    for (int i = -R; i <= R; ++i) { k[i + R] = std::exp(-0.5f * (i * i) / (sigma * sigma)); ksum += k[i + R]; }
    for (size_t i = 0; i < k.size(); ++i) k[i] /= ksum;      // energy-normalised
    std::vector<float> tmp(in.size(), 0.0f);
    out.assign(in.size(), 0.0f);
    for (int y = 0; y < H; ++y)                               // horizontal
        for (int x = 0; x < W; ++x) {
            float a = 0.0f;
            for (int i = -R; i <= R; ++i) {
                int xx = x + i; xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
                a += k[i + R] * in[static_cast<size_t>(y) * W + xx];
            }
            tmp[static_cast<size_t>(y) * W + x] = a;
        }
    for (int y = 0; y < H; ++y)                               // vertical
        for (int x = 0; x < W; ++x) {
            float a = 0.0f;
            for (int i = -R; i <= R; ++i) {
                int yy = y + i; yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
                a += k[i + R] * tmp[static_cast<size_t>(yy) * W + x];
            }
            out[static_cast<size_t>(y) * W + x] = a;
        }
}

// The AH weighting under test is the SHIPPING constant, speakcore::kHalWeight —
// deliberately not a private copy, so that changing the core's weights re-runs
// this measurement instead of silently drifting from it. (Its provenance is
// documented at the constant: the MECHANISM — light reaching the base is already
// depleted of blue/green by the yellow filter and upper layers, and the red
// layer sits closest to the base — is published; the {1, 0.30, 0.10} ratio is a
// MODELLED DEFAULT. An earlier comment here called them "Beer-Lambert-shaped
// (published AH behaviour)" with no citation, which fitted a made-up ratio and
// then cited Beer-Lambert for it; withdrawn.)
static const float kHalThresh = 0.6f;     // linear highlight excess threshold
static const float kHalSigma  = 9.0f;     // scatter radius (px, prototype)

struct Img { std::vector<float> p; int W, H; };   // interleaved RGB linear

// Build a scene: a bright disc (the source) on a dark blue-ish background.
// The disc colour is now a parameter — the old generator hard-coded a near-white
// (1.0, 0.92, 0.80) disc, which is exactly why no chromatic defect was visible.
static Img makeScene(int W, int H, float discR, float discG, float discB)
{
    Img im; im.W = W; im.H = H; im.p.assign(static_cast<size_t>(W) * H * 3, 0.0f);
    const float cx = W * 0.5f, cy = H * 0.5f, rad = W * 0.10f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            const size_t i = (static_cast<size_t>(y) * W + x) * 3;
            if (d <= rad) { im.p[i + 0] = discR; im.p[i + 1] = discG; im.p[i + 2] = discB; }
            else          { im.p[i + 0] = 0.010f; im.p[i + 1] = 0.014f; im.p[i + 2] = 0.030f; }
        }
    return im;
}
static Img makeSceneWhite(int W, int H, float v)   // the arm's original scene
{
    return makeScene(W, H, v, v * 0.92f, v * 0.80f);
}

// ---------------------------------------------------------------------------
// The two scatter SOURCE models under test.
//
// PER-CHANNEL (the subject): each channel's own light scatters, and the AH
// weight attenuates that channel's OWN scattered light.
//   S_c = w_c * blur(max(0, lin_c - t))
// A blue source has no red light to reflect, so it grows no red halo.
//
// MEAN-TINT (a NAMED control arm — this is what the prototype originally did):
// collapse RGB to a luminous mean, threshold that, and TINT the one achromatic
// field by w. This is physically incoherent: it lets BLUE photons manufacture a
// RED halo. Measured (source (0,0,2.0), mean 0.667 > 0.6): red scatter is 10x
// the blue scatter. It is kept here so H3 has something to fail against.
// ---------------------------------------------------------------------------
static void scatterPerChannel(const Img& im, float sigma, std::vector<float> sc[3])
{
    for (int c = 0; c < 3; ++c) {
        std::vector<float> ex(static_cast<size_t>(im.W) * im.H, 0.0f);
        for (size_t k = 0; k < ex.size(); ++k) {
            const float l = im.p[k * 3 + c];
            ex[k] = l > kHalThresh ? (l - kHalThresh) : 0.0f;
        }
        blurSep(ex, im.W, im.H, sigma, sc[c]);
    }
}
static void scatterMeanTint(const Img& im, float sigma, std::vector<float>& sc)
{
    std::vector<float> ex(static_cast<size_t>(im.W) * im.H, 0.0f);
    for (size_t k = 0; k < ex.size(); ++k) {
        const float m = (im.p[k * 3 + 0] + im.p[k * 3 + 1] + im.p[k * 3 + 2]) * (1.0f / 3.0f);
        ex[k] = m > kHalThresh ? (m - kHalThresh) : 0.0f;
    }
    blurSep(ex, im.W, im.H, sigma, sc);
}

// SUBJECT: per-channel scatter re-injected as EXPOSURE, before the curve.
static Img renderPhysical(const Img& im, const SpeakProfile& p, float amount, const float* w)
{
    std::vector<float> sc[3]; scatterPerChannel(im, kHalSigma, sc);
    Img o; o.W = im.W; o.H = im.H; o.p.assign(im.p.size(), 0.0f);
    for (size_t k = 0; k < sc[0].size(); ++k)
        for (int c = 0; c < 3; ++c) {
            const float E = im.p[k * 3 + c] + amount * w[c] * sc[c][k];   // RE-EXPOSURE
            o.p[k * 3 + c] = toneChannel(E, c, p);                        // then the curve
        }
    return o;
}

// CONTROL ARM (the old model): mean-then-tint scatter, same injection point.
static Img renderMeanTint(const Img& im, const SpeakProfile& p, float amount, const float* w)
{
    std::vector<float> sc; scatterMeanTint(im, kHalSigma, sc);
    Img o; o.W = im.W; o.H = im.H; o.p.assign(im.p.size(), 0.0f);
    for (size_t k = 0; k < sc.size(); ++k)
        for (int c = 0; c < 3; ++c)
            o.p[k * 3 + c] = toneChannel(im.p[k * 3 + c] + amount * w[c] * sc[k], c, p);
    return o;
}

// BASELINE: the same blurred excess, red-tinted, added to the OUTPUT.
static Img renderOverlay(const Img& im, const SpeakProfile& p, float amount, const float* w)
{
    std::vector<float> sc[3]; scatterPerChannel(im, kHalSigma, sc);
    Img o; o.W = im.W; o.H = im.H; o.p.assign(im.p.size(), 0.0f);
    for (size_t k = 0; k < sc[0].size(); ++k)
        for (int c = 0; c < 3; ++c)
            o.p[k * 3 + c] = toneChannel(im.p[k * 3 + c], c, p) + amount * w[c] * sc[c][k];
    return o;
}

// REFERENCE: no halation at all — the increment baseline H3 differences against.
static Img renderNone(const Img& im, const SpeakProfile& p)
{
    Img o; o.W = im.W; o.H = im.H; o.p.assign(im.p.size(), 0.0f);
    for (size_t k = 0; k < im.p.size() / 3; ++k)
        for (int c = 0; c < 3; ++c) o.p[k * 3 + c] = toneChannel(im.p[k * 3 + c], c, p);
    return o;
}

// Mean of a thin ring just outside the disc — that is where the halo lives. The
// window is scoped in units of SIGMA (not of the disc radius, as it was): the
// old fixed [1.15,1.6]*rad window was silently tuned to sigma=9 at W=220 and
// collapses at any other radius, which would make H1/H2 fail for reasons that
// have nothing to do with the physics.
static void ringMean(const Img& im, float& r, float& g, float& b)
{
    const float cx = im.W * 0.5f, cy = im.H * 0.5f, rad = im.W * 0.10f;
    const float lo = rad + 0.3f * kHalSigma, hi = rad + 2.0f * kHalSigma;
    double sr = 0, sg = 0, sb = 0; int n = 0;
    for (int y = 0; y < im.H; ++y)
        for (int x = 0; x < im.W; ++x) {
            const float d = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (d > lo && d < hi) {
                const size_t i = (static_cast<size_t>(y) * im.W + x) * 3;
                sr += im.p[i + 0]; sg += im.p[i + 1]; sb += im.p[i + 2]; n++;
            }
        }
    r = float(sr / n); g = float(sg / n); b = float(sb / n);
}
static float chromaOf(float r, float g, float b)
{
    float L, a, bb; dwgLinToLab(r, g, b, L, a, bb);
    return std::sqrt(a * a + bb * bb);
}

// The halo INCREMENT in a channel: how much the ring rose vs the no-halation
// reference. Differencing against the reference removes the background, which
// goes through the curve in every arm.
struct Ring { float r, g, b; };
static Ring ringOf(const Img& im) { Ring q; ringMean(im, q.r, q.g, q.b); return q; }

int main()
{
    printf("=== Speak Phase 4 prototype: halation vs a red overlay ===\n");
    const int W = 220, H = 220;
    SpeakProfile p = neutralProfile();
    const float amount = 1.0f;
    const float* w = speakcore::kHalWeight;   // the SHIPPING constant

    // --- the ceiling is ANALYTIC, not an empirical finding. State it as such. ---
    const float ceiling = toneChannel(1e6f, 0, p);
    const float printerOff = (p.printerMaster + p.printerLights[0]) * kPrinterPt;
    const float DprnMin = hdCurve(-p.negDmax[0] + printerOff, p.prnDmin[0], p.prnDmax[0],
                                  p.prnGamma[0], p.prnToe[0], p.prnShoulder[0], p.prnSpeed[0]);
    const float ceilClosed = k18Gray * pow10f(-(DprnMin - chainDensity(0.0f, 0, p)));
    const float bg = toneChannel(0.010f, 0, p);
    printf("  print paper-white ceiling = %.4f (closed form %.4f)  background floor = %.4f\n",
           ceiling, ceilClosed, bg);
    check(std::fabs(ceiling - ceilClosed) < 2e-3f,
          "P0 the paper-white ceiling matches its closed form (a PROPERTY, not a gate)",
          "sup toneChannel is bounded by hdCurve's algebra for ANY halation param");

    // ---------------------------------------------------------------- H1 / H2
    // These gate the INJECTION POINT: re-exposure vs an end-chain overlay.
    const float sources[5] = { 1.0f, 4.0f, 16.0f, 64.0f, 256.0f };
    float physRed[5], physChroma[5], ovRed[5], ovChroma[5];
    printf("  %-8s | %-22s | %-22s\n", "source", "physical (red, chroma)", "overlay  (red, chroma)");
    for (int i = 0; i < 5; ++i) {
        Img s = makeSceneWhite(W, H, sources[i]);
        Ring a = ringOf(renderPhysical(s, p, amount, w));
        physRed[i] = a.r; physChroma[i] = chromaOf(a.r, a.g, a.b);
        Ring d = ringOf(renderOverlay(s, p, amount, w));
        ovRed[i] = d.r; ovChroma[i] = chromaOf(d.r, d.g, d.b);
        printf("  %8.0f | %10.4f %10.1f  | %10.4f %10.1f\n",
               sources[i], physRed[i], physChroma[i], ovRed[i], ovChroma[i]);
    }

    check(physRed[4] <= ceiling * 1.02f, "H1a physical halo stays under paper white (analytic; see header)",
          std::to_string(physRed[4]) + " <= " + std::to_string(ceiling));
    check(ovRed[4] > ceiling * 1.5f, "H1b the red overlay blows straight past paper white (unbounded)",
          std::to_string(ovRed[4]) + " >> " + std::to_string(ceiling));

    float physPeak = 0.0f; int physPeakAt = 0;
    for (int i = 0; i < 5; ++i) if (physChroma[i] > physPeak) { physPeak = physChroma[i]; physPeakAt = i; }
    check(physPeakAt < 4 && physChroma[4] < physPeak,
          "H2a physical halo's chroma PEAKS then falls (it goes white-hot)",
          "peak at source " + std::to_string((int)sources[physPeakAt]) +
          ", falls to " + std::to_string(physChroma[4]));
    bool ovMono = true;
    for (int i = 1; i < 5; ++i) if (ovChroma[i] <= ovChroma[i - 1]) ovMono = false;
    check(ovMono, "H2b the overlay's chroma just keeps climbing (never desaturates)",
          "monotonic: " + std::string(ovMono ? "yes" : "no"));
    check(physChroma[4] < ovChroma[4], "H2c at extreme source the physical halo is far less saturated",
          std::to_string(physChroma[4]) + " < " + std::to_string(ovChroma[4]));

    // ------------------------------------------------------------------- H3
    // CHROMATIC FIDELITY — the gate the old arm could not see.
    printf("\n  --- H3 chromatic fidelity (the defect the white-disc arm was blind to) ---\n");

    // H3a: a WHITE source must grow a RED-DOMINANT halo. This is the module's
    // headline claim ("halation is the glow that says film"), and it is what
    // pins the weights: with w = {1,1,1} the ratio collapses to ~1.
    {
        Img s = makeSceneWhite(W, H, 16.0f);
        Ring ref = ringOf(renderNone(s, p));
        Ring hal = ringOf(renderPhysical(s, p, amount, w));
        const float dR = hal.r - ref.r, dB = hal.b - ref.b;
        const float ratio = dB > 1e-6f ? dR / dB : 1e9f;
        printf("  white source: halo increment  dR=%.4f  dB=%.4f  ->  R/B = %.2f\n", dR, dB, ratio);
        check(ratio > 5.0f, "H3a a white source's halo is RED-DOMINANT (pins the AH weighting)",
              "dR/dB = " + std::to_string(ratio) + " > 5");
    }

    // H3b: a saturated BLUE source must NOT grow a red halo. Blue light is what
    // the AH path absorbs most; there is no red light present to reflect. The
    // MEAN-TINT model manufactures one out of blue photons — that is the L3-class
    // bug (all four backends would have agreed on the same wrong answer, and
    // parity would have stayed green at 2e-5).
    {
        Img s = makeScene(W, H, 0.0f, 0.0f, 16.0f);      // pure blue highlight
        Ring ref = ringOf(renderNone(s, p));
        Ring pc  = ringOf(renderPhysical(s, p, amount, w));
        Ring mt  = ringOf(renderMeanTint(s, p, amount, w));
        const float dR_pc = pc.r - ref.r, dB_pc = pc.b - ref.b;
        const float dR_mt = mt.r - ref.r, dB_mt = mt.b - ref.b;
        printf("  blue source : per-channel dR=%.5f dB=%.5f | mean-tint dR=%.5f dB=%.5f\n",
               dR_pc, dB_pc, dR_mt, dB_mt);
        check(dR_pc < 0.02f * (dB_pc > 1e-6f ? dB_pc : 1.0f) || dR_pc < 1e-4f,
              "H3b per-channel: a BLUE source grows NO red halo",
              "dR = " + std::to_string(dR_pc));
        // The control arm must FAIL this — that is what makes H3b a real gate.
        check(dR_mt > 10.0f * (dR_pc + 1e-6f),
              "H3b' the MEAN-TINT control arm manufactures a red halo from blue light",
              "mean-tint dR = " + std::to_string(dR_mt) + " vs per-channel " + std::to_string(dR_pc));
    }

    // ------------------------------------------------------------------- H4
    // SENSITIVITY — an arm that passes when the physics is sabotaged is not an
    // arm. H3a must go RED when the AH weighting is deleted or inverted. (On the
    // OLD arm, measured: both sabotages passed 5/5.)
    printf("\n  --- H4 sensitivity: the arm must REJECT sabotaged AH weights ---\n");
    {
        const float wFlat[3] = { 1.0f, 1.0f, 1.0f };
        const float wInv[3]  = { 0.10f, 0.30f, 1.0f };
        Img s = makeSceneWhite(W, H, 16.0f);
        Ring ref = ringOf(renderNone(s, p));
        Ring flat = ringOf(renderPhysical(s, p, amount, wFlat));
        Ring inv  = ringOf(renderPhysical(s, p, amount, wInv));
        const float rFlat = (flat.r - ref.r) / (flat.b - ref.b);
        const float rInv  = (inv.r - ref.r) / (inv.b - ref.b);
        printf("  deleted  w={1,1,1}      -> halo R/B = %.2f\n", rFlat);
        printf("  inverted w={0.1,0.3,1}  -> halo R/B = %.2f\n", rInv);
        check(rFlat <= 5.0f, "H4a H3a REJECTS a deleted AH weighting (w={1,1,1})",
              "R/B = " + std::to_string(rFlat) + " <= 5");
        check(rInv <= 5.0f, "H4b H3a REJECTS an inverted (blue) AH weighting",
              "R/B = " + std::to_string(rInv) + " <= 5");
    }

    printf("\n%s (%d failures)\n", g_fail ? "PROTOTYPE REJECTED" : "PROTOTYPE EARNED ITS PLACE", g_fail);
    return g_fail ? 1 : 0;
}

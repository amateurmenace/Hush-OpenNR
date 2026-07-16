// Speak — shared parameter block passed from the OFX plugin to the GPU kernels.
//
// Speak is the film-reconstruction counterpart to Hush (see OpenNRPlugin.cpp):
// the LAST node in a grade, where Hush is the first. It lives in the SAME .ofx
// bundle as a second plugin (org.opennr.Speak) and inherits Hush's cardinal
// rule verbatim — speak_core.h is the single source of truth, and the three GPU
// kernels (SpeakMetalKernel.mm, SpeakCudaKernel.cu, SpeakOpenCLKernel.cpp) are
// line-by-line ports verified by a parity test (~2e-5 mean).
//
// EVERY member of BOTH structs below is 4 bytes, so the layout is identical in
// C++, Metal, CUDA and OpenCL with no padding on any of them. SpeakProfile is
// the engine's POD "look" struct — built-in stock families and user-calibrated
// profiles emit the SAME struct and hit exactly ONE kernel path. A field added
// here MUST also be added to the struct declarations inside SpeakMetalKernel.mm
// and SpeakOpenCLKernel.cpp kernel sources (CUDA includes this header directly),
// and the layout parity check in test/test_speak.cpp must be updated.
//
// MIT License.

#ifndef OPENNR_SPEAKPARAMS_H
#define OPENNR_SPEAKPARAMS_H

// ---------------------------------------------------------------------------
// SpeakProfile — the film "look", one versioned POD of 4-byte fields.
//
// The density spine is a cascaded two-stage tone scale:
//   scene logE  →  Negative H&D  →  Printer Lights  →  Print H&D  →  positive
// so there are TWO sets of per-channel Hurter-Driffield characteristic-curve
// handles (negative and print), with the printer lights injected as per-channel
// log-exposure offsets in the density gap between them (real lab points, where
// 1 point = 0.025 logE). Every handle is published-sensitometry-shaped; nothing
// clones a commercial profile.
//
// Curve handle semantics (per channel, both stages), all in log10 optical
// density on a log10-exposure axis (the physical sensitometric convention;
// the canonical cross-module datum is log2 stops and is converted internally):
//   Dmin      base/fog density (density at zero exposure)
//   Dmax      maximum density (the emulsion/paper ceiling)
//   gamma     straight-line contrast index dD/d(log10 E)
//   toe       toe knee sharpness  (larger = sharper toe,  ~0.1..40)
//   shoulder  shoulder knee sharpness (larger = sharper shoulder, ~0.1..40)
//   speed     speed point: log10-exposure (rel. 18% gray) where the
//             extrapolated straight line meets Dmin
// ---------------------------------------------------------------------------
typedef struct SpeakProfile
{
    // ---- Negative characteristic curves (R,G,B) ----
    float negDmin[3];
    float negDmax[3];
    float negGamma[3];
    float negToe[3];
    float negShoulder[3];
    float negSpeed[3];

    // ---- Printer lights: R/G/B color timing + master, in printer points ----
    // (1 point = 0.025 log10-exposure), injected in the negative→print gap.
    float printerLights[3];
    float printerMaster;

    // ---- Print characteristic curves (R,G,B) ----
    float prnDmin[3];
    float prnDmax[3];
    float prnGamma[3];
    float prnToe[3];
    float prnShoulder[3];
    float prnSpeed[3];

    // ---- Subtractive CMY dye coupling (Phase 2) ----
    float dyeCouple[9];   // inter-image coupler c_kj (row-major 3x3)
    float subSat[3];      // per-dye subtractive-saturation crosstalk strength
    float subSatKnee[3];  // per-dye Dmax knee in log-density

    // ---- Split toning / chromogenic crossover anchors (Phase 3) ----
    float splitShadow[3]; // CMY density offsets applied in the shadows
    float splitHigh[3];   // CMY density offsets applied in the highlights
    float splitPivot;     // tonal pivot (stops, rel. 18% gray) for the split
    float splitBalance;   // shadow/highlight weighting knob

    // ---- meta ----
    float systemGamma;    // target overall system gamma (~1.6), for the scope
    int   residualLUT;    // 0 = separable model only, 1 = 3D residual cube on
    int   profileVersion; // struct version for save/load compatibility
    int   _pad0;          // reserved (keeps the field count explicit)
} SpeakProfile;

// ---------------------------------------------------------------------------
// SpeakParams — the full per-render block: color-management + runtime state +
// the embedded look profile. Passed by value to the kernels exactly like
// Hush's NRParams. SpeakProfile is embedded LAST so its offsets stay stable as
// runtime fields are added ahead of it.
// ---------------------------------------------------------------------------

// inputColorSpace values
#define SPEAK_CS_DWG_INTERMEDIATE 0   // DaVinci Wide Gamut / Intermediate (default)
#define SPEAK_CS_REC709_G24       1   // Rec.709, gamma 2.4
#define SPEAK_CS_DWG_LINEAR       2   // DaVinci Wide Gamut, linear
#define SPEAK_CS_ACESCCT          3   // ACEScct
#define SPEAK_CS_LINEAR           4   // scene-linear passthrough (identity gamut)

// outputMode values
#define SPEAK_OUT_WORKING         0   // return DWG/DI, let RCM deliver (default)
#define SPEAK_OUT_BAKE_REC709     1   // apply DWG/DI -> Rec.709 as the last node

// viewMode values
#define SPEAK_VIEW_RESULT         0
#define SPEAK_VIEW_SPLIT          1   // input | result
#define SPEAK_VIEW_INPUT          2

typedef struct SpeakParams
{
    int   inputColorSpace; // SPEAK_CS_*
    int   outputMode;      // SPEAK_OUT_*
    int   grainRef;        // 0 display-referred, 1 scene-referred (Phase 4)
    float strength;        // 0..1 global look mix; 0 = bit-exact identity
    int   frameIndex;      // for deterministic grain / gate weave (Phase 4)
    int   viewMode;        // SPEAK_VIEW_*

    // ---- module enables (each stage's contribution is toggleable) ----
    int   enableTone;      // the density spine (negative -> printer -> print)
    int   enableDye;       // subtractive color (Phase 2)
    int   enableSplit;     // split toning (Phase 3)
    int   enableOptics;    // halation/bloom/grain/vignette (Phase 4)

    // ---- scopes (read-only, rendered INTO the image like Hush's) ----
    int   scopeHD;         // live H&D characteristic-curve scope
    int   scopeDensity;    // Status-M density waveform scope
    int   scopeVector;     // subtractive-sat vector field (Phase 2)

    SpeakProfile profile;
} SpeakParams;

#endif // OPENNR_SPEAKPARAMS_H

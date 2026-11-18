// A C interface over the zonos3.cpp core: load once, then embed voices and synthesize many texts.
// The text is expected already normalized (zyphra_textnorm on the Python side).
//
// Copyright 2026 Subtitld contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef ZS_BUILDING
#define ZS_API __declspec(dllexport)
#else
#define ZS_API __declspec(dllimport)
#endif
#else
#define ZS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zs_ctx zs_ctx;

// "<interface version> <ZONOS3-Mini commit>", e.g. "1 a50e141...".
ZS_API const char * zs_version(void);

// CPU only. spk may be NULL (no cloning). threads: CPU threads of the backbone, decoder and speaker encoder (<= 0:
// ggml's default for the backbone and every logical processor for the others). Returns NULL on failure with the reason
// in err.
ZS_API zs_ctx * zs_load(const char * backbone, const char * zac, const char * spk, int threads,
                        char * err, int err_len);
ZS_API void zs_unload(zs_ctx * c);
ZS_API int zs_sample_rate(const zs_ctx * c);
ZS_API int zs_speaker_dim(const zs_ctx * c);

// pcm: mono float32 at 44100 or 16000 Hz. out: zs_speaker_dim floats. Returns 0 on success.
ZS_API int zs_embed(zs_ctx * c, const float * pcm, int64_t n, int sample_rate, float * out);

// Optional emotion/style steering (steer.gguf). Load after zs_load and before a request carries `controls`.
// Returns 0 on success, -1 with the reason in err.
ZS_API int zs_steer_load(zs_ctx * c, const char * path, char * err, int err_len);
// The loaded bundle's directions as a JSON array [{"id","set","name","hidden","axis"}]; "" when none is loaded.
// Writes at most out_len bytes; returns the JSON length (may exceed out_len).
ZS_API int zs_steer_controls(zs_ctx * c, char * out, int out_len);

typedef struct zs_request {
    const char * text;          // normalized UTF-8
    const char * language;      // server language code (languages.json "code"), e.g. "en_us"; picks the filler
    const float * embedding;    // zs_speaker_dim floats, or NULL (no speaker row)
    float cfg_scale;            // speaker CFG, 1 = off (LACI runs only without CFG)
    double speaking_rate;       // < 0: unknown bucket
    uint32_t seed;
    int max_frames;             // <= 0: the registry default
    // LACI: the alignment heads JSON of the language (assets/laci_heads), NULL = plain generation. It stops a take once
    // the text has been read (no run-on), and retakes skips and stalls.
    const char * laci_heads;
    int short_text;             // with laci_heads: one or two words get the pause marker + filler and are cut there
    int tail_cut;               // with laci_heads: cut at the first silence once the text is read (see zonos3_c.cpp)
    const volatile int * cancel; // polled every frame, nonzero stops
    // Emotion/style doses: JSON {"<direction id>": dose, ...} (e.g. {"emotions.happy": 4}), NULL/"" = no steering.
    // Needs zs_steer_load(); raw-space directions move the speaker embedding, hidden-space ones add to the bank.
    const char * controls;
    // Word-level controls: JSON [{"start": char, "end": char, "controls": {"<id>": dose}], in UTF-8 characters
    // of `text`; NULL/"" = the constant `controls` apply to the whole line. Needs zs_steer_load().
    const char * spans;
} zs_request;

// On success *out holds *n_out samples at zs_sample_rate (free with zs_free); returns 0, or -1 with err set.
// info (may be NULL) receives the LACI record as JSON ("" without LACI).
ZS_API int zs_synthesize(zs_ctx * c, const zs_request * req, float ** out, int64_t * n_out,
                         char * err, int err_len, char * info, int info_len);
ZS_API void zs_free(float * p);

#ifdef __cplusplus
}
#endif

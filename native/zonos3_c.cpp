// Copyright 2026 Subtitld contributors
// SPDX-License-Identifier: Apache-2.0
//
// Mirrors zonos3-cli --tts (src/main.cpp: do_tts and emit_output) without the HTTP server or text normalization.
// With LACI it runs one request the way the server's worker does (src/server/scheduler.cpp worker_loop, one slot, no
// CFG twin), and short texts get the server's filler (src/server/generate.cpp short_text_prompt).
#include "zonos3_c.h"

#include "zonos3.h"
#include "zonos3-generation.h"
#include "zonos3-laci.h"
#include "zonos3-sampler.h"
#include "zonos3-short-text.h"
#include "zonos3-span-follow.h"
#include "zonos3-steer.h"
#include "zonos3-text.h"
#include "zonos3-tokenizer.h"
#include "json.hpp"
#include "zac.h"
#include "spk-encoder.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct zs_ctx {
    zonos3_model model;
    zac_model zac;
    spk_model spk;
    bool have_spk = false;
    // LACI: one single-slot batch context with the alignment probe, built on first use.
    zonos3_batch_ctx bc;
    bool have_bc = false;
    int capacity = 0;
    std::map<std::string, zonos3_laci::LaciHeads> heads;  // by path
    std::unordered_map<int32_t, int> token_bytes;
    // Emotion/style steering bundle (steer.gguf), loaded on demand by zs_steer_load.
    zonos3_steer_bundle steer;
    bool have_steer = false;
    std::string steer_path;
};

static void set_err(char * err, int len, const std::string & msg) {
    if (err && len > 0) snprintf(err, (size_t) len, "%s", msg.c_str());
}

#ifndef ZS_UPSTREAM_COMMIT
#define ZS_UPSTREAM_COMMIT "unknown"
#endif

const char * zs_version(void) { return "1 " ZS_UPSTREAM_COMMIT; }

zs_ctx * zs_load(const char * backbone, const char * zac, const char * spk, int threads, char * err, int err_len) {
    const int use_gpu = 0;
    auto * c = new (std::nothrow) zs_ctx();
    if (!c) return nullptr;
    if (!zonos3_model_load(c->model, backbone, use_gpu != 0)) {
        set_err(err, err_len, std::string("cannot load backbone ") + backbone);
        delete c;
        return nullptr;
    }
    if (!zac_load(c->zac, zac, use_gpu != 0)) {
        set_err(err, err_len, std::string("cannot load decoder ") + zac);
        zac_free(c->zac);
        zonos3_model_free(c->model);
        delete c;
        return nullptr;
    }
    if (spk && *spk) {
        if (!spk_load(c->spk, spk, use_gpu != 0)) {
            set_err(err, err_len, std::string("cannot load speaker encoder ") + spk);
            spk_free(c->spk);
            zac_free(c->zac);
            zonos3_model_free(c->model);
            delete c;
            return nullptr;
        }
        c->have_spk = true;
    }
    if (threads > 0) {
        ggml_backend_cpu_set_n_threads(c->model.backend, threads);
        ggml_backend_cpu_set_n_threads(c->zac.backend, threads);
        if (c->have_spk) ggml_backend_cpu_set_n_threads(c->spk.backend_cpu, threads);
    }
    const auto & tok = c->model.tok;
    for (const auto & [sym, v] : tok.vocab) {
        int n = 0;
        for (unsigned char ch : sym) n += (ch & 0xC0) != 0x80;
        c->token_bytes[v + tok.offset] = n;
    }
    for (const auto & [content, v] : tok.added) c->token_bytes[v + tok.offset] = (int) content.size();
    return c;
}

void zs_unload(zs_ctx * c) {
    if (!c) return;
    if (c->have_bc) zonos3_batch_free(c->bc);
    if (c->have_spk) spk_free(c->spk);
    zac_free(c->zac);
    zonos3_model_free(c->model);
    delete c;
}

int zs_sample_rate(const zs_ctx * c) { return c->zac.sample_rate; }
int zs_speaker_dim(const zs_ctx * c) { return (int) c->model.hp.speaker_dim; }

int zs_embed(zs_ctx * c, const float * pcm, int64_t n, int sample_rate, float * out) {
    if (!c->have_spk) return -1;
    std::vector<float> emb;
    if (!spk_embed(c->spk, pcm, n, sample_rate, emb) || emb.size() != c->model.hp.speaker_dim) return -1;
    std::copy(emb.begin(), emb.end(), out);
    return 0;
}

// ---- Emotion / style steering (steer.gguf: zonos3-steer.h) ------------------

// UTF-8 characters (code points) of the text, for the character-rate schedule.
static int utf8_chars(const std::string & s) {
    int n = 0;
    for (unsigned char ch : s) n += (ch & 0xC0) != 0x80;
    return n;
}

// {"<direction id>": dose, ...} -> the map zonos3_steer_resolve takes.
static bool parse_controls(const char * json_text, std::map<std::string, double> & out, std::string & err) {
    if (!json_text || !*json_text) return true;
    const nlohmann::json j = nlohmann::json::parse(json_text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        err = "controls must be a JSON object of {\"direction.id\": dose}";
        return false;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (!it.value().is_number()) {
            err = "control '" + it.key() + "' is not a number";
            return false;
        }
        out[it.key()] = it.value().get<double>();
    }
    return true;
}

// Word-level controls: [{"start": char, "end": char, "controls": {"<id>": dose}}].
static bool parse_spans(const char * json_text, std::vector<zonos3_control_span> & out, std::string & err) {
    if (!json_text || !*json_text) return true;
    const nlohmann::json j = nlohmann::json::parse(json_text, nullptr, false);
    if (j.is_discarded() || !j.is_array()) {
        err = "spans must be a JSON array of {start, end, controls}";
        return false;
    }
    for (const auto & entry : j) {
        if (!entry.is_object()) {
            err = "a span must be an object";
            return false;
        }
        zonos3_control_span span;
        span.start = entry.value("start", 0);
        span.end = entry.value("end", 0);
        if (span.start < 0 || span.end <= span.start) {
            err = "a span needs start >= 0 and end > start";
            return false;
        }
        const auto controls = entry.find("controls");
        if (controls == entry.end() || !controls->is_object()) {
            err = "a span needs a controls object";
            return false;
        }
        for (auto it = controls->begin(); it != controls->end(); ++it) {
            if (!it.value().is_number()) {
                err = "span control '" + it.key() + "' is not a number";
                return false;
            }
            span.controls[it.key()] = it.value().get<double>();
        }
        out.push_back(std::move(span));
    }
    return true;
}

// Word-level spans are written in characters; Chinese/Japanese/Korean characters hold far more speech than a
// Latin letter, so a 1:1 character-to-frame rate would end every span early. Weight by the CJK share of the line.
static double span_frames_per_char(const std::string & text) {
    size_t total = 0, cjk = 0;
    for (size_t i = 0; i < text.size();) {
        const unsigned char b = (unsigned char) text[i];
        if (b < 0x80) i += 1;
        else if ((b & 0xE0) == 0xC0) i += 2;
        else if ((b & 0xF0) == 0xE0) {
            if (b >= 0xE3 && b <= 0xEA) ++cjk;  // U+3000..U+9FFF / Hangul
            i += 3;
        } else i += 4;
        ++total;
    }
    if (!total) return 1.0;
    return 1.0 + 2.5 * ((double) cjk / (double) total);
}

int zs_steer_load(zs_ctx * c, const char * path, char * err, int err_len) {
    if (!c || !path || !*path) return -1;
    if (c->have_steer && c->steer_path == path) return 0;
    std::string e;
    zonos3_steer_bundle b;
    if (!zonos3_steer_load(b, path, c->model, e)) {
        set_err(err, err_len, e);
        return -1;
    }
    // The batch context owns the steering sources: rebuild it with the bundle's layout.
    if (c->have_bc) {
        zonos3_batch_free(c->bc);
        c->have_bc = false;
    }
    c->steer = std::move(b);
    c->steer_path = path;
    c->have_steer = true;
    return 0;
}

int zs_steer_controls(zs_ctx * c, char * out, int out_len) {
    if (!c || !c->have_steer) {
        if (out && out_len > 0) out[0] = 0;
        return 0;
    }
    const std::string s = zonos3_steer_controls_json(c->steer, (int) c->model.hp.speaker_dim);
    if (out && out_len > 0) snprintf(out, (size_t) out_len, "%s", s.c_str());
    return (int) s.size();
}

// The character range of `text` behind each id of its text span (server/generate.cpp text_token_ranges).
static std::vector<zonos3_follow::Range> text_token_ranges(const zs_ctx * c, const std::u32string & text) {
    const zonos3_tokenizer & tok = c->model.tok;
    zonos3_follow::BpeOffsets bpe;
    if (!tok.vocab.empty()) {
        bpe = [c, &tok](const std::u32string & piece) {
            std::vector<zonos3_follow::Range> out;
            const std::string utf8 = utf8_from_u32(piece);
            std::vector<int> char_of(utf8.size() + 1, (int) piece.size());
            for (size_t b = 0, ch = 0; b < utf8.size(); ++b) {
                if (b > 0 && ((unsigned char) utf8[b] & 0xC0) != 0x80) ++ch;
                char_of[b] = (int) ch;
            }
            const std::vector<int32_t> ids = zonos3_tokenize(tok, utf8);  // [BOS] tokens [EOS]
            size_t at = 0;
            for (size_t i = 1; i + 1 < ids.size(); ++i) {
                const auto it = c->token_bytes.find(ids[i]);
                const size_t n = it == c->token_bytes.end() ? 0 : (size_t) it->second;
                if (n == 0) continue;
                out.push_back({char_of[at], char_of[at + n - 1] + 1});
                at += n;
            }
            return out;
        };
    }
    zonos3_follow::TextMarkers markers;
    if (tok.has_phonemes)
        markers.phonemes = zonos3_follow::PhonemeMarkers{utf8_to_u32(tok.phonemes.open_marker),
                                                         utf8_to_u32(tok.phonemes.close_marker)};
    markers.start = tok.start_id >= 0;
    markers.pauses = !tok.pause_ids.empty();
    return zonos3_follow::token_char_ranges(text, bpe, markers);
}

// server/generate.cpp short_text_prompt: a qualifying text gets the pause marker and its language's filler; returns the
// text-span index of the filler's first token, else nullopt (text unchanged).
static std::optional<int> short_text_prompt(const zs_ctx * c, const std::string & language, std::string & text) {
    const auto & tok = c->model.tok;
    const auto & reg = zonos3_short::registry();
    const uint32_t marker = utf8_to_u32(reg.pause_marker)[0];
    const int bucket = (int) marker - 0xE003;
    if (tok.vocab.empty() || bucket < 0 || bucket >= (int) tok.pause_ids.size()) return std::nullopt;
    const auto filler = reg.fillers.find(language);
    if (filler == reg.fillers.end()) return std::nullopt;
    const std::u32string t32 = utf8_to_u32(text);
    int words = 0;
    bool in_word = false;
    for (char32_t ch : t32) {
        if (ch == 0xE000 || ch == 0xE001 || (ch >= 0xE002 && ch <= 0xE00A)) return std::nullopt;  // markers
        const bool space = is_py_space(ch);
        words += !space && !in_word;
        in_word = !space;
    }
    if (words > reg.max_words) return std::nullopt;
    std::vector<int32_t> ids, filler_ids;
    std::string err;
    if (!zonos3_payload_ids(tok, text, ids, err) || !zonos3_payload_ids(tok, filler->second, filler_ids, err))
        return std::nullopt;
    if (ids.empty() || (int64_t) ids.size() * reg.max_ratio[1] > (int64_t) filler_ids.size() * reg.max_ratio[0])
        return std::nullopt;
    const std::string full = text + reg.pause_marker + filler->second;
    const std::u32string f32 = utf8_to_u32(full);
    const auto ranges = text_token_ranges(c, f32);
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [a, b] = ranges[i];
        if (a <= (int) t32.size() || a >= b) continue;
        bool all_space = true;
        for (int k = a; k < b; ++k) all_space = all_space && is_py_space(f32[k]);
        if (!all_space) {
            text = full;
            return (int) i;
        }
    }
    return std::nullopt;
}

static void set_windows(zonos3_slot & slot, const std::optional<zonos3_laci::Windows> & mask) {
    slot.probe.windows.clear();
    if (mask) for (const auto & w : *mask) slot.probe.windows.push_back({w.head.layer, w.head.head, w.lo, w.hi});
}

static std::unique_ptr<zonos3_sampler> new_sampler(const zonos3_model & model, zonos3_sampling sp, uint32_t seed) {
    sp.seed = seed;
    return std::make_unique<zonos3_sampler>(sp, model.hp);
}

// One request under LACI (server/scheduler.cpp: attach_probe + the worker loop for a single slot). Fills `codes` with the
// audio stream (silence-prefill rows first) and `eos_frame` with its EOS stream row (-1: none).
static bool run_laci(zs_ctx * c, const zonos3_prompt & prompt, const zonos3_laci::LaciHeads & heads,
                     std::optional<int> short_text, const zonos3_sampling & sp, int max_frames,
                     bool tail_cut, const volatile int * cancel, const zonos3_steering * steer,
                     std::vector<int32_t> & codes, int & eos_frame, std::string & info,
                     std::string & err) {
    const zonos3_model & model = c->model;
    const auto & hp = model.hp;
    const int ncb = (int) hp.n_codebooks, W = ncb + 1, av = (int) hp.audio_vocab;
    const int n0 = prompt.n_rows, P = prompt.prefill_frames;
    if (!c->have_bc) {
        c->capacity = std::min((int) hp.n_ctx_train, zonos3_generation().max_frames + 4096);
        zonos3_batch_features features;
        features.probe_max_text = std::min(2048, c->capacity);
        const std::vector<int> sources = c->have_steer ? c->steer.sources : std::vector<int>{};
        if (!zonos3_batch_init(c->bc, model, 1, c->capacity, sources, features)) {
            err = "batch init failed";
            return false;
        }
        c->have_bc = true;
    }
    if (n0 + max_frames > c->capacity) {
        err = "prompt and frame budget exceed the slot capacity";
        return false;
    }
    zonos3_align_probe probe;
    int start = -1, len = 0;
    if (!zonos3_locate_text_span(model, prompt.ids.data(), n0, start, len)) {
        err = "the prompt has no complete text span for the alignment probe";
        return false;
    }
    if (len > c->bc.features.probe_max_text) {
        err = "the text is too long for the alignment probe";
        return false;
    }
    for (const auto & h : heads.heads) probe.heads.push_back({h.layer, h.head});
    probe.text_start = start;
    probe.text_len = len;
    zonos3_laci::ProbeControl probe_ctl;
    probe_ctl.text_len = len;

    zonos3_laci::SessionParams p;
    p.heads = heads;
    for (int r = n0 - P; r < n0; ++r)
        p.prefill_rows.emplace_back(prompt.ids.begin() + (size_t) r * W, prompt.ids.begin() + (size_t) (r + 1) * W);
    p.seed = (int64_t) sp.seed;
    p.max_tokens = max_frames;
    p.n_codebooks = ncb;
    if (hp.n_cond > 0) p.condition_values = std::vector<int>(prompt.cond.buckets.begin(), prompt.cond.buckets.end());
    for (size_t i = 0; i < hp.cond_names.size() && i < hp.cond_edges.size(); ++i)
        if (hp.cond_names[i] == "speaking_rate" && !hp.cond_edges[i].empty()) {
            zonos3_laci::RateBuckets rb;
            rb.index = (int) i;
            rb.edges.assign(hp.cond_edges[i].begin(), hp.cond_edges[i].end());
            rb.n_buckets = (int) hp.cond_bucket_counts[i];
            p.rates = rb;
        }
    p.hold = true;
    p.short_text = short_text;
    zonos3_laci::LaciSession laci(std::move(p));

    std::vector<float> idf(prompt.ids.begin(), prompt.ids.end());
    zonos3_slot slot;
    slot.index = 0;
    slot.active = true;
    slot.n_past = n0;
    slot.prefill = P;
    slot.probe = probe;
    set_windows(slot, laci.probe_mask());
    auto smp = new_sampler(model, sp, sp.seed);
    int attempt_max = max_frames;
    bool readout_ready = false;
    std::vector<float> logits((size_t) av * ncb, 0.0f);
    if (!zonos3_batch_slot_prefill(c->bc, 0, idf.data(), n0, prompt.cond, logits.data())) {
        err = "prefill failed";
        return false;
    }
    codes = zonos3_prefill_codes(model, prompt);
    int frames_done = 0;
    auto push_rows = [&](zonos3_laci::RowRange r) {
        const auto & stream = laci.stream();
        for (int i = r.begin; i < r.end; ++i) codes.insert(codes.end(), stream[i].begin(), stream[i].begin() + ncb);
    };

    for (;;) {
        if (cancel && *cancel) {
            err = "cancelled";
            return false;
        }
        // sample one row from the current logits; the readout of the last decode step goes to the session first
        if (readout_ready) {
            readout_ready = false;
            laci.on_step(probe_ctl, slot.step, slot.readout.argmax, (double) slot.readout.centroid);
            set_windows(slot, probe_ctl.mask);
        }
        std::vector<int32_t> sampled;
        smp->stop_bias = (float) laci.stop_boost();
        zonos3_slot_sample(model, slot, *smp, logits.data(), sampled, attempt_max, {});
        std::optional<int> eos;
        if (slot.eos_frame >= 0) eos = std::max(0, slot.eos_frame - laci.attempt_start());
        laci.append(slot.next_ids, eos);
        if (probe_ctl.stop) slot.done = true;
        if (!slot.done) push_rows(laci.drain());

        if (slot.done) {
            const std::optional<zonos3_laci::Continuation> cont = laci.finish_attempt();
            if (!cont) break;
            // rollback (scheduler.cpp laci_resume): re-prefill the prompt without its prefill rows + the kept rows
            const int base = n0 - P;
            const auto & stream = laci.stream();
            std::vector<float> ids((size_t) (base + cont->n_rows) * W);
            std::copy_n(idf.begin(), (size_t) base * W, ids.begin());
            for (int r = 0; r < cont->n_rows; ++r)
                for (int k = 0; k < W; ++k) ids[(size_t) (base + r) * W + k] = (float) stream[r][k];
            zonos3_conditioning cond = prompt.cond;
            if (cont->condition_values) cond.buckets.assign(cont->condition_values->begin(), cont->condition_values->end());
            slot.done = false;
            slot.step = 0;
            slot.eos_frame = -1;
            slot.countdown = -1;
            slot.n_past = base + cont->n_rows;
            slot.prefill = cont->n_rows;
            slot.next_ids.clear();
            set_windows(slot, laci.probe_mask());
            probe_ctl.mask.reset();
            probe_ctl.stop = false;
            readout_ready = false;
            attempt_max = cont->max_tokens;
            smp = new_sampler(model, sp, (uint32_t) ((uint64_t) cont->seed.value_or(sp.seed) & 0xFFFFFFFFu));
            for (int r = P; r < cont->n_rows; ++r) smp->accept(std::vector<int>(stream[r].begin(), stream[r].begin() + ncb));
            smp->generated = 0;
            if (!zonos3_batch_slot_prefill(c->bc, 0, ids.data(), slot.n_past, cond, logits.data())) {
                err = "re-prefill failed";
                return false;
            }
            push_rows(laci.drain());
            continue;  // the next row comes from the prefill logits
        }

        if (steer) {
            // Rows generated so far, silence prefill excluded; `codes` is not the running stream here.
            steer->delta_at(frames_done, slot.steer);
        }
        if (!zonos3_batch_step(c->bc, {&slot})) {
            err = "decode step failed";
            return false;
        }
        ++frames_done;
        memcpy(logits.data(), zonos3_batch_slot_logits(c->bc, 0), (size_t) av * ncb * sizeof(float));
        readout_ready = true;
    }
    push_rows(laci.drain(/*final=*/true));
    const auto & end = laci.end();
    eos_frame = end && end->second ? *end->second : -1;
    info = laci.record().to_json();

    // Tail cut (ours, not upstream's): the release checkpoint has no pause tokens, so the server's short-text filler
    // never applies and a take may run on for tail_frames after the text is read. Apply ShortCut's rule with the
    // detector's done row (the text read for its share and the reading on EOS) as the point the "filler" was reached:
    // cut at the first silent frame from `lookback` frames before it, after at least `min_voiced` voiced frames.
    const auto done = laci.detector() ? laci.detector()->done_row() : std::nullopt;
    if (tail_cut && done) {
        zonos3_short::ShortCut cut(len - 1, ncb);
        const auto & stream = laci.stream();
        const int n_prefill = laci.n_prefill();
        const int last = eos_frame >= 0 ? std::min((int) stream.size(), eos_frame + ncb) : (int) stream.size();
        for (int r = n_prefill; r < last && !cut.decided(); ++r) {
            if (r == *done) cut.observe_reading(r - n_prefill, len);
            cut.observe_row(stream[r]);
        }
        const auto & d = cut.decided();
        if (d && d->first == "cut" && (eos_frame < 0 || n_prefill + d->second < eos_frame)) {
            info += "\n{\"tail_cut\": " + std::to_string(n_prefill + d->second) + ", \"was\": " + std::to_string(eos_frame) + "}";
            eos_frame = n_prefill + d->second;
        }
    }
    return true;
}

int zs_synthesize(zs_ctx * c, const zs_request * req, float ** out, int64_t * n_out, char * err, int err_len,
                  char * info, int info_len) {
    *out = nullptr;
    *n_out = 0;
    set_err(info, info_len, "");
    const zonos3_model & model = c->model;
    const int max_frames = req->max_frames > 0 ? req->max_frames : zonos3_generation().max_frames;
    const bool use_laci = req->laci_heads && *req->laci_heads && req->cfg_scale == 1.0f;

    try {
        std::string text = req->text;
        std::optional<int> short_text;
        if (use_laci && req->short_text && req->language) short_text = short_text_prompt(c, req->language, text);

        // Emotion/style steering: resolve the doses into a hidden delta. Raw-space directions move the speaker
        // embedding, so it is copied and may be rewritten in place.
        zonos3_steering steering;
        std::vector<float> steered_embedding;
        const float * embedding = req->embedding;
        const zonos3_steering * steer_ptr = nullptr;
        const bool want_steer = (req->controls && *req->controls) || (req->spans && *req->spans);
        if (want_steer) {
            if (!c->have_steer) {
                set_err(err, err_len, "controls were given but no steering bundle (steer.gguf) is loaded");
                return -1;
            }
            std::map<std::string, double> doses;
            std::vector<zonos3_control_span> spans;
            std::string ctl_err;
            if (!parse_controls(req->controls, doses, ctl_err)) {
                set_err(err, err_len, ctl_err);
                return -1;
            }
            if (!parse_spans(req->spans, spans, ctl_err)) {
                set_err(err, err_len, ctl_err);
                return -1;
            }
            if (req->embedding) steered_embedding.assign(req->embedding, req->embedding + model.hp.speaker_dim);
            if (!zonos3_steer_resolve(c->steer, doses, spans, utf8_chars(text), span_frames_per_char(std::string(text)),
                                      req->embedding ? &steered_embedding : nullptr, steering, ctl_err)) {
                set_err(err, err_len, ctl_err);
                return -1;
            }
            if (req->embedding) embedding = steered_embedding.data();
            steer_ptr = &steering;
        }

        zonos3_prompt_options opt;
        if (embedding) opt.speaker_embedding = embedding;
        if (req->speaking_rate >= 0) {
            opt.condition_buckets.assign(model.hp.cond_bucket_counts.begin(), model.hp.cond_bucket_counts.end());
            const int idx = zonos3_condition_index(model, "speaking_rate");
            if (idx >= 0) opt.condition_buckets[idx] = zonos3_condition_bucket(model, idx, req->speaking_rate);
        }
        zonos3_prompt prompt;
        std::string why;
        if (!zonos3_build_prompt(model, text, opt, prompt, &why)) {
            set_err(err, err_len, "build_prompt: " + why);
            return -1;
        }
        if ((int64_t) prompt.n_rows + max_frames > model.hp.n_ctx_train) {
            set_err(err, err_len, "prompt and frame budget exceed the model context");
            return -1;
        }
        zonos3_sampling sp;
        sp.seed = req->seed;
        const int ncb = (int) model.hp.n_codebooks, prefill = prompt.prefill_frames;
        std::vector<int32_t> all;
        int eos_frame = -1, total = 0;

        if (use_laci) {
            auto it = c->heads.find(req->laci_heads);
            if (it == c->heads.end()) it = c->heads.emplace(req->laci_heads, zonos3_laci::LaciHeads::load(req->laci_heads)).first;
            std::string record, e;
            if (!run_laci(c, prompt, it->second, short_text, sp, max_frames, req->tail_cut != 0, req->cancel,
                          steer_ptr, all, eos_frame, record, e)) {
                set_err(err, err_len, e);
                return -1;
            }
            set_err(info, info_len, record);
            total = (int) all.size() / ncb;
            if (total <= prefill) return 0;
        } else {
            std::vector<int32_t> codes;
            zonos3_frame_cb on_frame;
            const volatile int * cancel = req->cancel;
            if (cancel) on_frame = [cancel](int, const int32_t *, int) { return *cancel == 0; };
            const int n_frames = zonos3_generate(model, prompt, max_frames, sp, req->embedding ? req->cfg_scale : 1.0f,
                                                 codes, eos_frame, true, nullptr, on_frame, steer_ptr);
            if (n_frames < 0) {
                set_err(err, err_len, "generation failed");
                return -1;
            }
            if (cancel && *cancel) {
                set_err(err, err_len, "cancelled");
                return -1;
            }
            if (n_frames == 0) return 0;
            all = zonos3_prefill_codes(model, prompt);
            all.insert(all.end(), codes.begin(), codes.end());
            total = prefill + n_frames;
        }

        std::vector<float> audio;
        if (!zac_decode(c->zac, all.data(), total, ncb, eos_frame, audio)) {
            set_err(err, err_len, "decode failed");
            return -1;
        }
        const size_t drop = std::min(audio.size(), (size_t) prefill * zac_samples_per_frame(c->zac));
        audio.erase(audio.begin(), audio.begin() + drop);
        const float gain = zac_output_gain(c->zac, -1.0);
        for (float & v : audio) v *= gain;
        zonos3_soft_limit(audio.data(), audio.size());

        auto * buf = (float *) malloc(audio.size() * sizeof(float));
        if (!buf) return -1;
        memcpy(buf, audio.data(), audio.size() * sizeof(float));
        *out = buf;
        *n_out = (int64_t) audio.size();
        return 0;
    } catch (const std::exception & e) {
        set_err(err, err_len, e.what());
        return -1;
    }
}

void zs_free(float * p) { free(p); }

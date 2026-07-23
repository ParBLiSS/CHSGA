#pragma once

// csr_traceback.hpp — checkpointed traceback for ParSGA CSR alignment
//
// Forward DP uses the same parallel kernels as parallel_align_CSR (OpenMP
// deletion partition, CAS match/sub, cut-graph insertion).  Backpointers are
// recovered post-hoc from the computed score rows (del+match tie-breaking
// matches serial_align_CSR; insert BP uses the cut graph for the read base).
//
// Checkpointing: same pattern as ConSGA traceback — save row 0 and every k-th
// row, then walk backward recomputing segments with full backpointer grids.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "config.hpp"
#include "csr_align.hpp"

// ---------------------------------------------------------------------------
// UninitU32Array — raw uint32_t backpointer buffer without std::vector's
// default zero-initialization. Every element of bp_all / bp_block is
// overwritten before it is read, so the zero-fill was pure (and, on
// Opossum-scale graphs, dominant) waste; see TB_PROFILE data.
// ---------------------------------------------------------------------------
struct UninitU32Array {
    std::unique_ptr<uint32_t[]> buf;
    UninitU32Array() = default;
    explicit UninitU32Array(size_t n) : buf(new uint32_t[n]) {}
    UninitU32Array(UninitU32Array&&) noexcept = default;
    UninitU32Array& operator=(UninitU32Array&&) noexcept = default;
    uint32_t*       data()       noexcept { return buf.get(); }
    const uint32_t* data() const noexcept { return buf.get(); }
    uint32_t&       operator[](size_t i)       noexcept { return buf[i]; }
    const uint32_t& operator[](size_t i) const noexcept { return buf[i]; }
};

// ---------------------------------------------------------------------------
// TB_PROFILE — optional phase timers, mirrors traceback.hpp (ConSGA side).
// ParSGA's DP row (deletion+match+insertion) is fused inside one #pragma omp
// parallel region with internal barriers, so it is timed as a single unit
// rather than split into del/match/insert sub-phases (splitting further would
// require adding barrier-synchronized timers inside the parallel region).
// ---------------------------------------------------------------------------

#ifdef TB_PROFILE
#include <chrono>
#include <cstdio>

struct TBProfile {
    double bp_alloc_s      = 0.0; ///< allocating bp_all / bp_block
    double fwd_dp_s        = 0.0; ///< forward-pass DP row (del+match+insert fused)
    double ckpt_save_s     = 0.0;
    double ckpt_load_s     = 0.0;
    double seg_dp_s        = 0.0; ///< per-segment recompute DP row (del+match+insert fused)
    double traceback_walk_s = 0.0;
    double ckpt_erase_s    = 0.0;
    long   n_segments      = 0;

    void reset() { *this = TBProfile{}; }
};
inline TBProfile g_tb_prof;

struct TBScopeTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double* acc;
    explicit TBScopeTimer(double* a) : acc(a) {}
    ~TBScopeTimer() {
        *acc += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }
};
#define TB_TIME_BLOCK(acc_field) TBScopeTimer _tb_scope_timer(&(acc_field))
#else
#define TB_TIME_BLOCK(acc_field) do {} while (0)
#endif

// ---------------------------------------------------------------------------
// Backpointer encoding (aligns with ConSGA traceback.hpp)
// ---------------------------------------------------------------------------

static constexpr uint8_t ParBP_DELETE   = 0;
static constexpr uint8_t ParBP_MATCHSUB = 1;
static constexpr uint8_t ParBP_INSERT   = 2;

inline uint32_t par_bp_encode(uint8_t op, int32_t src) noexcept {
    return (uint32_t(op) << 30) | (uint32_t(src) & 0x3FFFFFFFu);
}
inline uint8_t par_bp_op(uint32_t enc) noexcept { return uint8_t(enc >> 30); }
inline int32_t par_bp_src(uint32_t enc) noexcept { return int32_t(enc & 0x3FFFFFFFu); }

struct ParAlignOp {
    int64_t row;
    int32_t vertex;
    uint8_t op;
};

struct ParCheckpointConfig {
    int64_t     k       = 500;
    std::string dir     = "/tmp/parsga_traceback_ckpts";
    bool        cleanup = true;
};

struct ParCheckpointManager {
    ParCheckpointConfig cfg;
    size_t              bytes_per_row;

    ParCheckpointManager(ParCheckpointConfig c, size_t bpr)
        : cfg(std::move(c)), bytes_per_row(bpr)
    {
        std::filesystem::create_directories(cfg.dir);
    }

    [[nodiscard]] std::string path_for(int64_t row) const {
        return cfg.dir + "/row_" + std::to_string(row) + ".bin";
    }

    template<typename LogScore>
    void save(int64_t row, const LogScore* data) const {
        std::string p = path_for(row);
        std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
        if (!ofs) throw std::runtime_error("checkpoint write failed: " + p);
        ofs.write(reinterpret_cast<const char*>(data),
                  static_cast<std::streamsize>(bytes_per_row));
        if (!ofs) throw std::runtime_error("checkpoint write incomplete: " + p);
    }

    template<typename LogScore>
    void load(int64_t row, LogScore* data) const {
        std::string p = path_for(row);
        std::ifstream ifs(p, std::ios::binary);
        if (!ifs) throw std::runtime_error("checkpoint read failed: " + p);
        ifs.read(reinterpret_cast<char*>(data),
                 static_cast<std::streamsize>(bytes_per_row));
        if (!ifs) throw std::runtime_error("checkpoint read incomplete: " + p);
    }

    void erase(int64_t row) const {
        std::filesystem::remove(path_for(row));
    }
};

// ---------------------------------------------------------------------------
// Parallel graph scan helpers
// ---------------------------------------------------------------------------

static inline void par_atomic_min_i32(int32_t* addr, int32_t val) {
    int32_t cur = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (val < cur
           && !__atomic_compare_exchange_n(
               addr, &cur, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
        ;
}

// Record insert predecessor when insert strictly beat del+match (smallest u wins).
template<typename LogScore>
inline void par_csr_try_insert_bp(
    uint32_t*           bp_row,
    const LogScore*     no_ins,
    int32_t             v,
    int32_t             u,
    LogScore            val)
{
    if (val >= no_ins[static_cast<size_t>(v)])
        return;
    const uint32_t enc  = bp_row[v];
    const uint8_t  op   = par_bp_op(enc);
    const int32_t  cur_u = par_bp_src(enc);
    if (op != ParBP_INSERT || u < cur_u)
        bp_row[v] = par_bp_encode(ParBP_INSERT, u);
}

// Cut-graph insertion with live backpointer recording (mirrors csr_align.hpp insertion).
template<typename LogScore>
inline void par_csr_insertion_with_bp(
    const int&                num_vertices,
    uint8_t&                  thread_num,
    const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE],
    int32_t                   read_base_index,
    const std::vector<int> (&component_indices)[ALPHABET_SIZE],
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    LogScore*                 edits,
    LogScore                  ins_score,
    const LogScore*           no_ins,
    uint32_t*                 bp_row)
{
    auto& tco      = thread_component_offsets[read_base_index];
    auto& comp_idx = component_indices[read_base_index];
    auto& offs     = cut_offsets_out[read_base_index];
    auto& adj      = cut_adjcny_out[read_base_index];

    int component_after_this_thread = -1;
    for (int i = 1; component_after_this_thread == -1; ++i)
        component_after_this_thread = tco[thread_num + i];

    if constexpr (dag) {
        for (int j = tco[thread_num];
             j != -1 && j < component_after_this_thread; ++j)
        {
            const int32_t  u     = comp_idx[j];
            const LogScore base_u =
                static_cast<LogScore>(edits[u] + ins_score);
            for (int32_t e = offs[u]; e < offs[u + 1]; ++e) {
                const int32_t  v    = adj[e];
                LogScore       prev = edits[v];
                const LogScore cand = base_u;
                if (cand < prev) {
                    LogScore upd = cand;
                    if (cas_update<LogScore>(&edits[v], prev, upd))
                        par_csr_try_insert_bp<LogScore>(
                            bp_row, no_ins, v, u, upd);
                } else if (cand == prev) {
                    par_csr_try_insert_bp<LogScore>(
                        bp_row, no_ins, v, u, cand);
                }
            }
        }
    } else {
        static thread_local std::vector<int>      queue;
        static thread_local std::vector<uint32_t> in_queue_epoch;
        static thread_local uint32_t              current_epoch = 1;

        if (in_queue_epoch.size() < static_cast<size_t>(num_vertices))
            in_queue_epoch.resize(static_cast<size_t>(num_vertices), 0);

        queue.clear();
        size_t qhead = 0;

        auto relax_from = [&](int32_t u) {
            const LogScore base_u =
                static_cast<LogScore>(edits[u] + ins_score);
            for (int32_t e = offs[u]; e < offs[u + 1]; ++e) {
                const int32_t  v    = adj[e];
                LogScore       prev = edits[v];
                const LogScore cand = base_u;
                if (cand < prev) {
                    LogScore upd = cand;
                    if (cas_update<LogScore>(&edits[v], prev, upd)) {
                        par_csr_try_insert_bp<LogScore>(
                            bp_row, no_ins, v, u, upd);
                        if (in_queue_epoch[static_cast<size_t>(v)]
                            != current_epoch)
                        {
                            in_queue_epoch[static_cast<size_t>(v)] =
                                current_epoch;
                            queue.push_back(v);
                        }
                    }
                } else if (cand == prev) {
                    par_csr_try_insert_bp<LogScore>(
                        bp_row, no_ins, v, u, cand);
                }
            }
        };

        for (int j = tco[thread_num];
             j != -1 && j < component_after_this_thread; ++j)
            relax_from(comp_idx[j]);

        while (qhead < queue.size()) {
            const int x = queue[qhead++];
            in_queue_epoch[static_cast<size_t>(x)] = 0;
            relax_from(x);
        }

        if (++current_epoch == 0) {
            std::fill(in_queue_epoch.begin(), in_queue_epoch.end(), 0);
            current_epoch = 1;
        }
    }
}

template<typename LogScore>
inline int32_t par_csr_parallel_best_vertex(
    const LogScore* row,
    int32_t         num_vertices)
{
    const int nthr = omp_get_max_threads();
    std::vector<LogScore> tbest(static_cast<size_t>(nthr));
    std::vector<int32_t>  tv(static_cast<size_t>(nthr));

    #pragma omp parallel
    {
        const int t = omp_get_thread_num();
        LogScore  lb = row[0];
        int32_t   lv = 0;

        #pragma omp for schedule(static)
        for (int32_t v = 0; v < num_vertices; ++v) {
            if (row[v] < lb || (row[v] == lb && v < lv)) {
                lb = row[v];
                lv = v;
            }
        }

        tbest[static_cast<size_t>(t)] = lb;
        tv[static_cast<size_t>(t)]    = lv;
    }

    LogScore  best_score = row[0];
    int32_t   best_v     = 0;
    for (int t = 0; t < nthr; ++t) {
        if (tbest[static_cast<size_t>(t)] < best_score
            || (tbest[static_cast<size_t>(t)] == best_score
                && tv[static_cast<size_t>(t)] < best_v))
        {
            best_score = tbest[static_cast<size_t>(t)];
            best_v     = tv[static_cast<size_t>(t)];
        }
    }
    return best_v;
}

// ---------------------------------------------------------------------------
// Parallel DP helpers (same kernels as parallel_align_CSR)
// ---------------------------------------------------------------------------

inline std::vector<int> par_csr_vertex_thread_offsets(int32_t num_vertices) {
    const int nthr = omp_get_max_threads();
    std::vector<int> off(static_cast<size_t>(nthr) + 1);
    for (int t = 0; t <= nthr; ++t)
        off[static_cast<size_t>(t)] =
            get_thread_offset(t, nthr, num_vertices);
    return off;
}

template<typename LogScore>
inline void par_csr_parallel_init_row(
    LogScore*                 row0,
    int32_t                   num_vertices,
    const std::vector<char>&  vertex_labels,
    char                      read_base,
    LogScore                  match_score,
    LogScore                  subst_score)
{
    #pragma omp parallel for
    for (int32_t v = 0; v < num_vertices; ++v) {
        row0[v] = match_sub<LogScore>(
            vertex_labels[static_cast<size_t>(v)],
            read_base, match_score, subst_score);
    }
}

template<typename LogScore>
inline void par_csr_parallel_dp_row(
    LogScore*                 edits[2],
    bool                      src,
    bool                      dst,
    char                      read_base,
    int32_t                   num_vertices,
    const std::vector<char>&  vertex_labels,
    const std::vector<int>&   offsets_out,
    const std::vector<int>&   edges,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    const std::vector<int> (&component_indices)[ALPHABET_SIZE],
    const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE],
    const std::vector<int>&   vertex_parallel_thread_offsets,
    LogScore                  match_score,
    LogScore                  subst_score,
    LogScore                  del_score,
    LogScore                  ins_score)
{
    #pragma omp parallel
    {
        uint8_t thread_num = static_cast<uint8_t>(omp_get_thread_num());
        const int t0 = vertex_parallel_thread_offsets[static_cast<size_t>(thread_num)];
        const int t1 = vertex_parallel_thread_offsets[static_cast<size_t>(thread_num) + 1];

        common_deletion_code<LogScore>(
            edits, src, dst, del_score, t0, t1);
        #pragma omp barrier

        vertex_parallel_match_mismatch_CSR<LogScore>(
            num_vertices, vertex_labels, offsets_out, edges,
            edits, read_base, src, dst, match_score, subst_score);
        #pragma omp barrier

        const int32_t read_base_index = base_index(read_base);
        #if ALPHABET_SIZE == 4
        if (read_base_index != 4)
        #endif
        {
            insertion<LogScore>(
                num_vertices, thread_num, thread_component_offsets,
                read_base_index, component_indices,
                cut_offsets_out, cut_adjcny_out,
                edits[dst], ins_score);
        }
        #pragma omp barrier
    }
}

template<typename LogScore>
inline void par_csr_vertex_parallel_match_with_bp(
    int32_t                   num_vertices,
    const std::vector<char>&  vertex_labels,
    const std::vector<int>&   offsets_out,
    const std::vector<int>&   edges,
    LogScore*                 edits[2],
    char                      read_base,
    bool                      src,
    bool                      dst,
    LogScore                  match_score,
    LogScore                  subst_score,
    uint32_t*                 bp_row)
{
    #pragma omp for schedule(static)
    for (int32_t u = 0; u < num_vertices; ++u) {
        for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
            const int32_t v = edges[e];
            LogScore prev = edits[dst][v];
            LogScore upd  = static_cast<LogScore>(
                edits[src][u]
                + match_sub<LogScore>(
                    vertex_labels[static_cast<size_t>(v)],
                    read_base, match_score, subst_score));
            if (cas_update<LogScore>(&edits[dst][v], prev, upd))
                bp_row[v] = par_bp_encode(ParBP_MATCHSUB, u);
        }
    }
}

// Insert backpointers only — call after parallel del+match (no_ins = pre-insert row).
template<typename LogScore>
inline void par_csr_compute_insert_row_bp(
    const LogScore*           dp_dst,
    const LogScore*           no_ins,
    uint32_t*                 bp_row,
    char                      read_base,
    int32_t                   num_vertices,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    LogScore                  ins_score)
{
    const int32_t rbi = base_index(read_base);
    #if ALPHABET_SIZE == 4
    if (rbi == 4)
        return;
    #endif

    const auto& offs = cut_offsets_out[rbi];
    const auto& adj  = cut_adjcny_out[rbi];
    std::vector<int32_t> insert_src(
        static_cast<size_t>(num_vertices), std::numeric_limits<int32_t>::max());

    #pragma omp parallel for schedule(static)
    for (int32_t u = 0; u < num_vertices; ++u) {
        const LogScore du = static_cast<LogScore>(dp_dst[u] + ins_score);
        for (int32_t e = offs[u]; e < offs[u + 1]; ++e) {
            const int32_t v = adj[e];
            if (dp_dst[v] != no_ins[static_cast<size_t>(v)])
                continue;
            if (dp_dst[v] != du)
                continue;
            par_atomic_min_i32(&insert_src[static_cast<size_t>(v)], u);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int32_t v = 0; v < num_vertices; ++v) {
        const int32_t u = insert_src[static_cast<size_t>(v)];
        if (u != std::numeric_limits<int32_t>::max())
            bp_row[v] = par_bp_encode(ParBP_INSERT, u);
    }
}

// One DP row with backpointers recorded in the same pass (no duplicate edge scan).
template<typename LogScore>
inline void par_csr_parallel_dp_row_with_bp(
    LogScore*                 edits[2],
    bool                      src,
    bool                      dst,
    char                      read_base,
    int32_t                   num_vertices,
    const std::vector<char>&  vertex_labels,
    const std::vector<int>&   offsets_out,
    const std::vector<int>&   edges,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    const std::vector<int> (&component_indices)[ALPHABET_SIZE],
    const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE],
    const std::vector<int>&   vertex_parallel_thread_offsets,
    LogScore                  match_score,
    LogScore                  subst_score,
    LogScore                  del_score,
    LogScore                  ins_score,
    uint32_t*                 bp_row,
    LogScore*                 no_ins_scratch)
{
    #pragma omp parallel
    {
        uint8_t thread_num = static_cast<uint8_t>(omp_get_thread_num());
        const int t0 = vertex_parallel_thread_offsets[static_cast<size_t>(thread_num)];
        const int t1 = vertex_parallel_thread_offsets[static_cast<size_t>(thread_num) + 1];

        common_deletion_code<LogScore>(
            edits, src, dst, del_score, t0, t1);
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int32_t v = 0; v < num_vertices; ++v)
            bp_row[v] = par_bp_encode(ParBP_DELETE, v);

        par_csr_vertex_parallel_match_with_bp<LogScore>(
            num_vertices, vertex_labels, offsets_out, edges,
            edits, read_base, src, dst, match_score, subst_score, bp_row);
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int32_t v = 0; v < num_vertices; ++v)
            no_ins_scratch[v] = edits[dst][v];

        const int32_t read_base_index = base_index(read_base);
        #if ALPHABET_SIZE == 4
        if (read_base_index != 4)
        #endif
        {
            par_csr_insertion_with_bp<LogScore>(
                num_vertices, thread_num, thread_component_offsets,
                read_base_index, component_indices,
                cut_offsets_out, cut_adjcny_out,
                edits[dst], ins_score, no_ins_scratch, bp_row);
        }
        #pragma omp barrier
    }
}

// Post-hoc backpointers for one DP row: parallel del+match (CAS), then
// parallel cut-graph insert scan with smallest-predecessor tie-breaking.
template<typename LogScore>
inline void par_csr_compute_row_bp(
    const LogScore*           dp_src,
    const LogScore*           dp_dst,
    uint32_t*                 bp_row,
    char                      read_base,
    int32_t                   num_vertices,
    const std::vector<char>&  vertex_labels,
    const std::vector<int>&   offsets_out,
    const std::vector<int>&   edges,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    LogScore                  match_score,
    LogScore                  subst_score,
    LogScore                  del_score,
    LogScore                  ins_score)
{
    std::vector<LogScore> no_ins(static_cast<size_t>(num_vertices));
    #pragma omp parallel for schedule(static)
    for (int32_t v = 0; v < num_vertices; ++v) {
        no_ins[static_cast<size_t>(v)] =
            static_cast<LogScore>(dp_src[v] + del_score);
        bp_row[v] = par_bp_encode(ParBP_DELETE, v);
    }

    #pragma omp parallel for schedule(static)
    for (int32_t u = 0; u < num_vertices; ++u) {
        for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
            const int32_t v = edges[e];
            const LogScore cand = static_cast<LogScore>(
                dp_src[u]
                + match_sub<LogScore>(
                    vertex_labels[static_cast<size_t>(v)],
                    read_base, match_score, subst_score));
            LogScore prev = no_ins[static_cast<size_t>(v)];
            LogScore upd  = cand;
            if (cas_update<LogScore>(
                    &no_ins[static_cast<size_t>(v)], prev, upd))
            {
                bp_row[v] = par_bp_encode(ParBP_MATCHSUB, u);
            }
        }
    }

    const int32_t rbi = base_index(read_base);
    #if ALPHABET_SIZE == 4
    if (rbi != 4)
    #endif
    {
        const auto& offs = cut_offsets_out[rbi];
        const auto& adj  = cut_adjcny_out[rbi];
        std::vector<int32_t> insert_src(
            static_cast<size_t>(num_vertices), std::numeric_limits<int32_t>::max());

        #pragma omp parallel for schedule(static)
        for (int32_t u = 0; u < num_vertices; ++u) {
            const LogScore du = static_cast<LogScore>(dp_dst[u] + ins_score);
            for (int32_t e = offs[u]; e < offs[u + 1]; ++e) {
                const int32_t v = adj[e];
                if (dp_dst[v] != no_ins[static_cast<size_t>(v)])
                    continue;
                if (dp_dst[v] != du)
                    continue;
                par_atomic_min_i32(
                    &insert_src[static_cast<size_t>(v)], u);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int32_t v = 0; v < num_vertices; ++v) {
            const int32_t u = insert_src[static_cast<size_t>(v)];
            if (u != std::numeric_limits<int32_t>::max())
                bp_row[v] = par_bp_encode(ParBP_INSERT, u);
        }
    }
}

// Deletions + match/substitution with backpointers (serial; segment verify).
// edges u → v (same loop structure as serial_align_CSR).
template<typename LogScore>
void par_csr_fused_del_sub_with_bp(
    LogScore*                 dp_dst,
    uint32_t*                 bp_row,
    const LogScore*           dp_src,
    const std::vector<char>&  vertex_labels,
    const std::vector<int>&   offsets_out,
    const std::vector<int>&   edges,
    char                      read_base,
    LogScore                  match_score,
    LogScore                  subst_score,
    LogScore                  del_score,
    int32_t                   num_vertices)
{
    for (int32_t v = 0; v < num_vertices; ++v) {
        dp_dst[v] = static_cast<LogScore>(dp_src[v] + del_score);
        bp_row[v] = par_bp_encode(ParBP_DELETE, v);
    }

    for (int32_t u = 0; u < num_vertices; ++u) {
        for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
            const int32_t v = edges[e];
            const LogScore cand = static_cast<LogScore>(
                dp_src[u]
                + match_sub<LogScore>(
                    vertex_labels[v],
                    read_base,
                    match_score,
                    subst_score));
            if (cand < dp_dst[v]) {
                dp_dst[v] = cand;
                bp_row[v] = par_bp_encode(ParBP_MATCHSUB, u);
            }
        }
    }
}

// After full-graph insertion (serial_align_CSR semantics), record insertion
// predecessors. Smallest predecessor u wins on ties.
template<typename LogScore>
void par_csr_compute_insert_bp(
    const LogScore*         dp_row,
    uint32_t*               bp_row,
    int32_t                 num_vertices,
    const std::vector<int>& offsets_out,
    const std::vector<int>& edges,
    LogScore                ins_score)
{
    std::vector<int32_t> insert_src(
        static_cast<size_t>(num_vertices), std::numeric_limits<int32_t>::max());

    #pragma omp parallel for schedule(static)
    for (int32_t u = 0; u < num_vertices; ++u) {
        const LogScore du = static_cast<LogScore>(dp_row[u] + ins_score);
        for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
            const int32_t v = edges[e];
            if (dp_row[v] == du)
                par_atomic_min_i32(&insert_src[static_cast<size_t>(v)], u);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int32_t v = 0; v < num_vertices; ++v) {
        const int32_t u = insert_src[static_cast<size_t>(v)];
        if (u != std::numeric_limits<int32_t>::max())
            bp_row[v] = par_bp_encode(ParBP_INSERT, u);
    }
}

// Full-graph insertion relaxations (matches serial_align_CSR insertion layer).
template<typename LogScore>
void par_csr_insertion_full_graph(
    LogScore*               dp_row,
    int32_t                 num_vertices,
    const std::vector<int>& offsets_out,
    const std::vector<int>& edges,
    LogScore                ins_score)
{
    if constexpr (dag) {
        for (int32_t u = 0; u < num_vertices; ++u) {
            for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
                const int32_t v = edges[e];
                dp_row[v]       = std::min(
                    dp_row[v],
                    static_cast<LogScore>(dp_row[u] + ins_score));
            }
        }
    } else {
        std::vector<char> in_queue(num_vertices, 0);
        std::vector<int>  q;
        q.reserve(num_vertices);

        for (int32_t u = 0; u < num_vertices; ++u) {
            const LogScore base_u = dp_row[u] + ins_score;
            for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
                const int32_t v = edges[e];
                if (base_u < dp_row[v]) {
                    dp_row[v] = base_u;
                    if (!in_queue[v]) {
                        in_queue[v] = 1;
                        q.push_back(v);
                    }
                }
            }
        }

        for (size_t qi = 0; qi < q.size(); ++qi) {
            const int    x     = q[qi];
            in_queue[x]        = 0;
            const LogScore base_x = dp_row[x] + ins_score;
            for (int32_t e = offsets_out[x]; e < offsets_out[x + 1]; ++e) {
                const int32_t v = edges[e];
                if (base_x < dp_row[v]) {
                    dp_row[v] = base_x;
                    if (!in_queue[v]) {
                        in_queue[v] = 1;
                        q.push_back(v);
                    }
                }
            }
        }
    }
}

template<typename LogScore>
void par_csr_recompute_segment_with_bp(
    const std::string&       read,
    int64_t                  seg_start,
    int64_t                  seg_end,
    const LogScore*          checkpoint_row,
    uint32_t*                bp_block,
    int32_t                  num_vertices,
    const std::vector<char>& vertex_labels,
    const std::vector<int>& offsets_out,
    const std::vector<int>& edges,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    const std::vector<int> (&component_indices)[ALPHABET_SIZE],
    const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE],
    const std::vector<int>&  vertex_parallel_thread_offsets,
    LogScore                 match_score,
    LogScore                 subst_score,
    LogScore                 del_score,
    LogScore                 ins_score)
{
    const size_t cols = static_cast<size_t>(num_vertices);
    std::vector<LogScore> buf(2 * cols);
    LogScore* edits[2] = { buf.data(), buf.data() + cols };

    std::memcpy(edits[0], checkpoint_row, cols * sizeof(LogScore));
    std::vector<LogScore> no_ins(cols);

    for (int64_t i = seg_start + 1; i <= seg_end; ++i) {
        const bool     src = ((i - 1) & 1) != 0;
        const bool     dst = (i & 1) != 0;
        const size_t   local  = static_cast<size_t>(i - seg_start - 1);
        uint32_t*      bp_row = bp_block + local * cols;
        const char     read_base = read[static_cast<size_t>(i)];

        {
            TB_TIME_BLOCK(g_tb_prof.seg_dp_s);
            par_csr_parallel_dp_row_with_bp<LogScore>(
                edits, src, dst, read_base, num_vertices,
                vertex_labels, offsets_out, edges,
                cut_offsets_out, cut_adjcny_out,
                component_indices, thread_component_offsets,
                vertex_parallel_thread_offsets,
                match_score, subst_score, del_score, ins_score,
                bp_row, no_ins.data());
        }
    }
}

[[nodiscard]] inline int32_t par_csr_traceback_segment(
    int64_t               seg_start,
    int64_t               seg_end,
    int32_t               end_vertex,
    const uint32_t*       bp_block,
    int32_t               num_vertices,
    std::vector<ParAlignOp>& path)
{
    const int32_t cols = num_vertices;
    int64_t       row  = seg_end;
    int32_t       v    = end_vertex;

    while (row > seg_start) {
        const size_t local = static_cast<size_t>(row - seg_start - 1);
        const uint32_t enc = bp_block[local * static_cast<size_t>(cols) +
                                       static_cast<size_t>(v)];
        const uint8_t op = par_bp_op(enc);
        const int32_t src = par_bp_src(enc);

        path.push_back({row, v, op});

        if (op == ParBP_INSERT) {
            v = src;
        } else {
            v = src;
            --row;
        }
    }
    return v;
}

template<typename LogScore>
std::vector<ParAlignOp> par_csr_align_read_with_traceback(
    const std::string&       read,
    int32_t                  num_vertices,
    const std::vector<char>& vertex_labels,
    const std::vector<int>& offsets_out,
    const std::vector<int>& edges,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    const std::vector<int> (&component_indices)[ALPHABET_SIZE],
    const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE],
    LogScore                 match_score,
    LogScore                 subst_score,
    LogScore                 del_score,
    LogScore                 ins_score,
    const ParCheckpointConfig& cfg)
{
    const int64_t read_len  = static_cast<int64_t>(read.size());
    const int64_t k         = cfg.k;
    const size_t  cols      = static_cast<size_t>(num_vertices);
    const size_t  row_bytes = cols * sizeof(LogScore);
    const size_t  bp_store_bytes =
        static_cast<size_t>(read_len) * cols * sizeof(uint32_t);
#ifndef FAST_BP_MAX_BYTES_OVERRIDE
    constexpr size_t FAST_BP_MAX_BYTES = 400ULL * 1024 * 1024 * 1024;
#else
    constexpr size_t FAST_BP_MAX_BYTES = FAST_BP_MAX_BYTES_OVERRIDE;
#endif

    const std::vector<int> vertex_parallel_thread_offsets =
        par_csr_vertex_thread_offsets(num_vertices);

#ifdef TB_PROFILE
    g_tb_prof.reset();
    const auto _tb_read_t0 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "TBPROF decision read_len=%lld num_chars=%zu bp_store_GB=%.2f cap_GB=%.2f path=%s\n",
        static_cast<long long>(read_len), cols,
        static_cast<double>(bp_store_bytes) / (1024.0*1024.0*1024.0),
        static_cast<double>(FAST_BP_MAX_BYTES) / (1024.0*1024.0*1024.0),
        (bp_store_bytes <= FAST_BP_MAX_BYTES) ? "FAST" : "CHECKPOINTED");
#endif

    if (bp_store_bytes <= FAST_BP_MAX_BYTES) {
        std::vector<LogScore> edit_array(2 * cols);
        LogScore* edits[2] = { edit_array.data(), edit_array.data() + cols };
        UninitU32Array bp_all;
        {
            TB_TIME_BLOCK(g_tb_prof.bp_alloc_s);
            bp_all = UninitU32Array(static_cast<size_t>(read_len) * cols);
        }

        std::vector<LogScore> no_ins(cols);

        par_csr_parallel_init_row<LogScore>(
            edits[0], num_vertices, vertex_labels, read[0],
            match_score, subst_score);

        for (int64_t i = 1; i < read_len; ++i) {
            const bool src = ((i - 1) & 1) != 0;
            const bool dst = (i & 1) != 0;
            uint32_t* bp_row =
                bp_all.data() + static_cast<size_t>(i) * cols;

            {
                TB_TIME_BLOCK(g_tb_prof.fwd_dp_s);
                par_csr_parallel_dp_row_with_bp<LogScore>(
                    edits, src, dst, read[static_cast<size_t>(i)], num_vertices,
                    vertex_labels, offsets_out, edges,
                    cut_offsets_out, cut_adjcny_out,
                    component_indices, thread_component_offsets,
                    vertex_parallel_thread_offsets,
                    match_score, subst_score, del_score, ins_score,
                    bp_row, no_ins.data());
            }
        }

        const int64_t   last     = read_len - 1;
        LogScore* const last_row = edits[static_cast<int>(last & 1)];
        const int32_t   best_vertex =
            par_csr_parallel_best_vertex<LogScore>(last_row, num_vertices);

        std::vector<ParAlignOp> path;
        path.reserve(static_cast<size_t>(read_len) * 2);

        {
            TB_TIME_BLOCK(g_tb_prof.traceback_walk_s);
            int32_t cur_vertex = best_vertex;
            int64_t cur_row    = read_len - 1;
            while (cur_row > 0) {
                const uint32_t enc =
                    bp_all[static_cast<size_t>(cur_row) * cols
                           + static_cast<size_t>(cur_vertex)];
                const uint8_t op  = par_bp_op(enc);
                const int32_t src = par_bp_src(enc);

                path.push_back({cur_row, cur_vertex, op});

                if (op == ParBP_INSERT) {
                    cur_vertex = src;
                } else {
                    cur_vertex = src;
                    --cur_row;
                }
            }

            path.push_back({0, cur_vertex, ParBP_MATCHSUB});
            std::reverse(path.begin(), path.end());
        }
#ifdef TB_PROFILE
        {
            const double total_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - _tb_read_t0).count();
            std::fprintf(stderr,
                "TBPROF FAST bp_alloc_s=%.3f fwd_dp_s=%.3f walk_s=%.3f total_s=%.3f\n",
                g_tb_prof.bp_alloc_s, g_tb_prof.fwd_dp_s,
                g_tb_prof.traceback_walk_s, total_s);
        }
#endif
        return path;
    }

    ParCheckpointManager ckpt(cfg, row_bytes);
    int32_t best_vertex = 0;

    {
        std::vector<LogScore> edit_array(2 * cols);
        LogScore* edits[2] = { edit_array.data(), edit_array.data() + cols };

        par_csr_parallel_init_row<LogScore>(
            edits[0], num_vertices, vertex_labels, read[0],
            match_score, subst_score);
        {
            TB_TIME_BLOCK(g_tb_prof.ckpt_save_s);
            ckpt.save(0, edits[0]);
        }

        for (int64_t i = 1; i < read_len; ++i) {
            const bool src = ((i - 1) & 1) != 0;
            const bool dst = (i & 1) != 0;
            {
                TB_TIME_BLOCK(g_tb_prof.fwd_dp_s);
                par_csr_parallel_dp_row<LogScore>(
                    edits, src, dst, read[static_cast<size_t>(i)], num_vertices,
                    vertex_labels, offsets_out, edges,
                    cut_offsets_out, cut_adjcny_out,
                    component_indices, thread_component_offsets,
                    vertex_parallel_thread_offsets,
                    match_score, subst_score, del_score, ins_score);
            }

            if (i % k == 0) {
                TB_TIME_BLOCK(g_tb_prof.ckpt_save_s);
                ckpt.save(i, edits[dst]);
            }
        }

        const int64_t   last     = read_len - 1;
        LogScore* const last_row = edits[static_cast<int>(last & 1)];
        best_vertex = par_csr_parallel_best_vertex<LogScore>(last_row, num_vertices);
    }

    // No segment spans more than min(k, read_len) rows (see traceback.hpp for
    // the argument); this halves the allocation for short reads on huge
    // graphs, and UninitU32Array skips the zero-init every cell gets
    // overwritten before it's read anyway.
    const int64_t  bp_block_rows = std::min<int64_t>(k, read_len);
    UninitU32Array bp_block(static_cast<size_t>(bp_block_rows) * cols);
    std::vector<LogScore> ckpt_row(cols);
    std::vector<ParAlignOp> path;
    path.reserve(static_cast<size_t>(read_len) * 2);

    int32_t cur_vertex = best_vertex;
    int64_t cur_row    = read_len - 1;

    while (cur_row > 0) {
        int64_t seg_start = (cur_row % k == 0) ? (cur_row - k) : (cur_row / k * k);
        if (seg_start < 0) seg_start = 0;
        const int64_t seg_end = cur_row;

        {
            TB_TIME_BLOCK(g_tb_prof.ckpt_load_s);
            ckpt.load(seg_start, ckpt_row.data());
        }

        par_csr_recompute_segment_with_bp<LogScore>(
            read, seg_start, seg_end, ckpt_row.data(), bp_block.data(),
            num_vertices, vertex_labels, offsets_out, edges,
            cut_offsets_out, cut_adjcny_out,
            component_indices, thread_component_offsets,
            vertex_parallel_thread_offsets,
            match_score, subst_score, del_score, ins_score);

        {
            TB_TIME_BLOCK(g_tb_prof.traceback_walk_s);
            cur_vertex = par_csr_traceback_segment(
                seg_start, seg_end, cur_vertex, bp_block.data(), num_vertices, path);
        }

#ifdef TB_PROFILE
        ++g_tb_prof.n_segments;
#endif

        if (cfg.cleanup && seg_start > 0) {
            TB_TIME_BLOCK(g_tb_prof.ckpt_erase_s);
            ckpt.erase(seg_start);
        }

        cur_row = seg_start;
    }

    path.push_back({0, cur_vertex, ParBP_MATCHSUB});
    if (cfg.cleanup) {
        TB_TIME_BLOCK(g_tb_prof.ckpt_erase_s);
        ckpt.erase(0);
    }

    std::reverse(path.begin(), path.end());

#ifdef TB_PROFILE
    {
        const double total_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - _tb_read_t0).count();
        std::fprintf(stderr,
            "TBPROF CHECKPOINTED fwd_dp_s=%.3f ckpt_save_s=%.3f ckpt_load_s=%.3f "
            "seg_dp_s=%.3f traceback_walk_s=%.3f ckpt_erase_s=%.3f n_segments=%ld total_s=%.3f\n",
            g_tb_prof.fwd_dp_s, g_tb_prof.ckpt_save_s, g_tb_prof.ckpt_load_s,
            g_tb_prof.seg_dp_s, g_tb_prof.traceback_walk_s, g_tb_prof.ckpt_erase_s,
            g_tb_prof.n_segments, total_s);
    }
#endif
    return path;
}

template<typename LogScore>
std::vector<std::vector<ParAlignOp>> par_csr_align_reads_with_traceback(
    const std::vector<std::string>& reads,
    int32_t                         num_vertices,
    const std::vector<char>&        vertex_labels,
    const std::vector<int>&         offsets_out,
    const std::vector<int>&        edges,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    const std::vector<int> (&component_indices)[ALPHABET_SIZE],
    const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE],
    LogScore                       match_score,
    LogScore                       subst_score,
    LogScore                       del_score,
    LogScore                       ins_score,
    const ParCheckpointConfig&     cfg)
{
    std::vector<std::vector<ParAlignOp>> out;
    out.reserve(reads.size());
    for (size_t ri = 0; ri < reads.size(); ++ri) {
        ParCheckpointConfig rcfg = cfg;
        rcfg.dir = cfg.dir + "/read_" + std::to_string(ri);
        out.push_back(par_csr_align_read_with_traceback<LogScore>(
            reads[ri], num_vertices, vertex_labels, offsets_out, edges,
            cut_offsets_out, cut_adjcny_out, component_indices,
            thread_component_offsets,
            match_score, subst_score, del_score, ins_score, rcfg));
    }
    return out;
}

[[nodiscard]] inline std::string par_ops_to_cigar(
    const std::vector<ParAlignOp>& path)
{
    std::string cigar;
    if (path.empty()) return cigar;

    auto op_char = [](uint8_t op) -> char {
        switch (op) {
            case ParBP_DELETE:   return 'D';
            case ParBP_MATCHSUB: return 'M';
            default:             return 'I';
        }
    };

    char prev_c = op_char(path[0].op);
    int  run    = 1;

    for (size_t i = 1; i < path.size(); ++i) {
        const char c = op_char(path[i].op);
        if (c == prev_c) {
            ++run;
        } else {
            cigar += std::to_string(run);
            cigar += prev_c;
            prev_c = c;
            run    = 1;
        }
    }
    cigar += std::to_string(run);
    cigar += prev_c;
    return cigar;
}

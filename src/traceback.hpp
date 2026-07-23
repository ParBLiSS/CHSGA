#pragma once

// ============================================================================
// traceback.hpp  —  Checkpointed traceback for ConSGA alignment
//
// Problem scale
//   num_chars (columns) : 800 000 000
//   readLength (rows)   : up to 1 000 000
//   Machine             : dual AMD EPYC 9534, 3 TB DDR5, 4 TB NVMe
//
// Strategy
//   Forward pass  – identical DP to align_read_combo; save every k-th row
//                   to NVMe.  Only 2 rows are resident in RAM at any time.
//   Traceback     – working from the last row backward, one segment at a
//                   time:
//                     1. Load the nearest checkpoint (row seg_start) from NVMe.
//                     2. Recompute rows [seg_start+1 … seg_end], storing a
//                        uint32_t backpointer per cell.
//                     3. Follow backpointers backward from (seg_end, best_v)
//                        to (seg_start, landing_v).
//                     4. Repeat for the previous segment.
//
// Memory (int16_t scores, 800 M columns, k = 500)
//   Forward working rows  :   2 × 1.6 GB  =    3.2 GB  (RAM)
//   Checkpoint files      :  ≤2000 × 1.6 GB =  3.2 TB  (NVMe)
//   Backpointer block     :  500 × 3.2 GB  =    1.6 TB  (RAM, one segment)
//   Recompute score rows  :   2 × 1.6 GB   =    3.2 GB  (RAM, two-row buf)
//
// Backpointer encoding
//   Each cell is one uint32_t: top 2 bits = operation, bottom 30 bits =
//   source vertex.  NodeId is uint32_t; 800 M < 2^30 = 1.07 B, so 30 bits
//   suffice.
//
//   BP_DELETE   (0): came from (row-1, same vertex v)  – skip a read base
//   BP_MATCHSUB (1): came from (row-1, predecessor u)  – align base to graph
//   BP_INSERT   (2): came from (row,   predecessor u)  – traverse graph edge
//
// Insert backpointers
//   insertionQueryLayerParallelization uses PCH shortcuts that do not store
//   middle vertices, so we cannot expand them during traceback.  Instead, we
//   run insertion_func_parallel normally (correct, fast scores), then do a
//   single O(V + E) post-hoc scan of the ORIGINAL string-graph edges: for
//   every edge u→v, if dp[v_first] == dp[u_last] + ins_score then v_first
//   was reachable in one hop from u_last; that is a valid traceback
//   predecessor.  PCH shortcuts are equivalent to chains of original edges, so
//   at least one original edge on every shortcut path satisfies the equality.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "align.hpp"   // init_func_parallel, fused_deletion_substitution,
                       // insertion_func_parallel, InCSR, match_sub, etc.

// ============================================================================
// UninitU32Array  —  raw uint32_t backpointer buffer without the
// zero-initialization that std::vector<uint32_t>(n) performs.
//
// Every element of bp_all / bp_block is overwritten (by fused_del_sub_with_bp
// or compute_insert_bp) before it is ever read, so std::vector's default
// value-initialization is pure waste — and on Opossum-scale graphs it is the
// *dominant* cost: profiling showed ~64% of per-read checkpointed-traceback
// time was spent zero-filling a buffer that gets fully overwritten a moment
// later. new[] without parentheses leaves POD elements uninitialized.
// ============================================================================
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

// Raw POD buffer without value-initialization (same rationale as UninitU32Array).
template<typename T>
struct UninitArray {
    std::unique_ptr<T[]> buf;
    UninitArray() = default;
    explicit UninitArray(size_t n) : buf(new T[n]) {}
    UninitArray(UninitArray&&) noexcept = default;
    UninitArray& operator=(UninitArray&&) noexcept = default;
    T*       data()       noexcept { return buf.get(); }
    const T* data() const noexcept { return buf.get(); }
    T&       operator[](size_t i)       noexcept { return buf[i]; }
    const T& operator[](size_t i) const noexcept { return buf[i]; }
};

// Build an in-CSR from a const Graph (align.hpp's build_in_csr takes Graph&&).
inline InCSR build_in_csr_from(const Graph& G) {
    InCSR in;
    in.in_offset.assign(G.n + 1, 0);
    for (NodeId u = 0; u < G.n; ++u)
        for (EdgeId e = G.offset[u]; e < G.offset[u + 1]; ++e)
            ++in.in_offset[G.E[e].v + 1];
    for (NodeId v = 0; v < G.n; ++v)
        in.in_offset[v + 1] += in.in_offset[v];
    in.in_E.resize(in.in_offset[G.n]);
    std::vector<size_t> cur = in.in_offset;
    for (NodeId u = 0; u < G.n; ++u)
        for (EdgeId e = G.offset[u]; e < G.offset[u + 1]; ++e) {
            NodeId v = G.E[e].v;
            in.in_E[cur[v]++] = InCSR::Edge{u};
        }
    return in;
}

// ============================================================================
// TB_PROFILE  —  optional fine-grained phase timers (opt-in via -DTB_PROFILE)
//
// Optional fine-grained phase timers (deletion/substitution, PCH insertion,
// insert-backpointer scan, NVMe checkpoint I/O) for the fast path and the
// checkpointed path. Zero overhead when TB_PROFILE is not defined.
// ============================================================================

#ifdef TB_PROFILE
#include <chrono>
#include <cstdio>

struct TBProfile {
    double bp_alloc_s      = 0.0; ///< allocating bp_all / bp_block (incl. any value-init)
    double fwd_dp_s        = 0.0; ///< forward-pass fused_del_sub_with_bp (fast path) or fused_deletion_substitution (checkpointed forward pass)
    double fwd_ins_s       = 0.0; ///< forward-pass insertion_func_parallel
    double fwd_insbp_s     = 0.0; ///< forward-pass compute_insert_bp (fast path only)
    double ckpt_save_s     = 0.0; ///< NVMe checkpoint writes (checkpointed path)
    double ckpt_load_s     = 0.0; ///< NVMe checkpoint reads (checkpointed path)
    double seg_dp_s        = 0.0; ///< per-segment recompute: fused_del_sub_with_bp
    double seg_ins_s       = 0.0; ///< per-segment recompute: insertion_func_parallel
    double seg_insbp_s     = 0.0; ///< per-segment recompute: compute_insert_bp
    double traceback_walk_s = 0.0; ///< following backpointers (fast path or within a segment)
    double ckpt_erase_s    = 0.0; ///< deleting spent checkpoint files
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

// ============================================================================
// Backpointer encoding / decoding
// ============================================================================

static constexpr uint8_t BP_DELETE   = 0;
static constexpr uint8_t BP_MATCHSUB = 1;
static constexpr uint8_t BP_INSERT   = 2;

static_assert(sizeof(NodeId) == 4,
    "NodeId must be uint32_t so 30-bit src field covers 800 M vertices");

inline uint32_t bp_encode(uint8_t op, NodeId src) noexcept {
    return (uint32_t(op) << 30) | (src & 0x3FFFFFFFu);
}
inline uint8_t bp_op (uint32_t enc) noexcept { return uint8_t(enc >> 30); }
inline NodeId  bp_src(uint32_t enc) noexcept { return NodeId(enc & 0x3FFFFFFFu); }

// ============================================================================
// One step in the recovered alignment path
// ============================================================================

struct AlignOp {
    int64_t row;    ///< 0-indexed read position; INSERT keeps the same row
    NodeId  vertex; ///< char-position in the graph label array
    uint8_t op;     ///< BP_DELETE / BP_MATCHSUB / BP_INSERT
};

// ============================================================================
// Configuration for checkpointing
// ============================================================================

struct CheckpointConfig {
    int64_t     k       = 500;                ///< save a checkpoint every k rows
    std::string dir     = "/tmp/consga_ckpt"; ///< NVMe directory for files
    bool        cleanup = true;               ///< delete checkpoint files after use
};

// ============================================================================
// Checkpoint I/O helper
// ============================================================================

struct CheckpointManager {
    CheckpointConfig cfg;
    size_t           bytes_per_row; ///< num_chars * sizeof(LogScore)

    CheckpointManager(CheckpointConfig c, size_t bpr)
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

// ============================================================================
// fused_del_sub_with_bp
//
// Identical DP to fused_deletion_substitution, but also writes one
// uint32_t backpointer per cell (op + source vertex).
// ============================================================================

template<typename LogScore>
void fused_del_sub_with_bp(
        LogScore*                 dp_dst,
        uint32_t*                 bp_row,
        const LogScore*           dp_src,
        const std::vector<char>&  vertex_labels,
        const InCSR&              inG,
        char                      read_base,
        LogScore                  match_score,
        LogScore                  subst_score,
        LogScore                  del_score,
        size_t                    num_chars)
{
    constexpr size_t BLK = 4096;
    parlay::blocked_for(0, num_chars, BLK,
        [&](size_t /*b*/, size_t lo, size_t hi) {
            for (size_t v = lo; v < hi; ++v) {
                // Deletion: advance read, stay at same graph vertex
                LogScore best   = dp_src[v] + del_score;
                NodeId   src_v  = static_cast<NodeId>(v);
                uint8_t  src_op = BP_DELETE;

                // Match / substitution: advance read AND traverse one graph edge
                const LogScore delta = match_sub<LogScore>(
                        vertex_labels[v], read_base, match_score, subst_score);
                for (size_t ie = inG.in_offset[v]; ie < inG.in_offset[v + 1]; ++ie) {
                    NodeId   u    = inG.in_E[ie].u;
                    LogScore cand = dp_src[u] + delta;
                    if (cand < best) {
                        best   = cand;
                        src_v  = u;
                        src_op = BP_MATCHSUB;
                    }
                }

                dp_dst[v] = best;
                bp_row[v] = bp_encode(src_op, src_v);
            }
        });
}

// ============================================================================
// compute_insert_bp  —  post-hoc insertion backpointer pass
//
// Must be called AFTER insertion_func_parallel has updated dp_row[].
//
// For each edge u→v in the original string graph, if
//   dp_row[v_first] == dp_row[u_last] + ins_score
// then v_first was reached in one graph hop from u_last (any PCH shortcut
// decomposes to a chain of such hops, each satisfying the equality), so we
// record (BP_INSERT, u_last) for v_first.
//
// For UseVertexLabels=true, also handle within-vertex char positions.
// ============================================================================

template<typename LogScore, typename Mapper>
void compute_insert_bp(
        const LogScore*  dp_row,
        uint32_t*        bp_row,
        const Graph&     origin_str_graph,
        const Mapper&    M,
        LogScore         ins_score)
{
    // 1) Within-vertex positions (only when vertex labels span >1 char)
    if constexpr (Mapper::hasLabels) {
        parlay::parallel_for(0, origin_str_graph.n, [&](NodeId u) {
            for (NodeId p = M.first(u); p < M.last(u); ++p) {
                if (dp_row[p + 1] == dp_row[p] + ins_score) {
                    bp_row[p + 1] = bp_encode(BP_INSERT, p);
                }
            }
        });
    }

    // 2) Across-vertex edges in the original (non-contracted) string graph
    parlay::parallel_for(0, origin_str_graph.n, [&](NodeId u) {
        const NodeId u_last = M.last(u);
        for (EdgeId e = origin_str_graph.offset[u];
             e < origin_str_graph.offset[u + 1]; ++e)
        {
            NodeId v_first = M.first(origin_str_graph.E[e].v);
            if (dp_row[v_first] == dp_row[u_last] + ins_score) {
                bp_row[v_first] = bp_encode(BP_INSERT, u_last);
            }
        }
    });
}

// ============================================================================
// recompute_segment_with_bp
//
// Recomputes rows [seg_start+1 … seg_end] starting from checkpoint_row
// (which holds the scores at row seg_start).  For each recomputed row i,
// the backpointers are stored at
//
//   bp_block[ (i - seg_start - 1) * num_chars + v ]
//
// This uses the same fast PCH insertion path as the forward pass, then
// runs compute_insert_bp to fill in the INSERT backpointers.
// ============================================================================

template<typename LogScore, typename Mapper>
void recompute_segment_with_bp(
        const std::string&        read,
        int64_t                   seg_start,
        int64_t                   seg_end,       // inclusive
        const LogScore*           checkpoint_row,
        uint32_t*                 bp_block,      // size >= (seg_end-seg_start)*num_chars
        size_t                    num_chars,
        const std::vector<char>&  vertex_labels_char,
        const InCSR&              in_char_graph,
        const Graph&              origin_str_graph,
        const sequence<NodeId>&   contracted_str_to_og,
        LogScore                  match_score,
        LogScore                  subst_score,
        LogScore                  del_score,
        LogScore                  ins_score,
        PchQuery&                 query,
        const Mapper&             Mstr)
{
    // Two-row score buffer; reused for the whole segment
    std::vector<LogScore> buf(2 * num_chars);
    LogScore* prev_dp = buf.data();
    LogScore* curr_dp = buf.data() + num_chars;

    std::memcpy(prev_dp, checkpoint_row, num_chars * sizeof(LogScore));

    for (int64_t i = seg_start + 1; i <= seg_end; ++i) {
        const size_t  local  = static_cast<size_t>(i - seg_start - 1);
        uint32_t*     bp_row = bp_block + local * num_chars;

        // Phase A: deletions + substitutions (writes dp + backpointers)
        {
            TB_TIME_BLOCK(g_tb_prof.seg_dp_s);
            fused_del_sub_with_bp<LogScore>(
                curr_dp, bp_row,
                prev_dp,
                vertex_labels_char,
                in_char_graph,
                read[static_cast<size_t>(i)],
                match_score, subst_score, del_score,
                num_chars);
        }

        // Phase B: insertions via PCH (updates curr_dp, exact same path as
        //          the forward pass → identical scores)
        {
            TB_TIME_BLOCK(g_tb_prof.seg_ins_s);
            insertion_func_parallel<LogScore, Mapper>(
                curr_dp,
                origin_str_graph,
                contracted_str_to_og,
                ins_score,
                query,
                Mstr);
        }

        // Phase C: post-hoc insert backpointers (overwrites bp_row entries
        //          where insertion won over deletion/substitution)
        {
            TB_TIME_BLOCK(g_tb_prof.seg_insbp_s);
            compute_insert_bp<LogScore, Mapper>(
                curr_dp, bp_row,
                origin_str_graph, Mstr,
                ins_score);
        }

        std::swap(prev_dp, curr_dp);
    }
}

// ============================================================================
// traceback_segment
//
// Traces backward from (seg_end, end_vertex) to (seg_start, ?), appending
// AlignOps in reverse order (the caller reverses the whole path at the end).
//
// INSERT ops stay at the same row and follow the insertion chain.
// DELETE / MATCHSUB ops advance to the previous row.
//
// Returns the vertex reached at row seg_start.
// ============================================================================

[[nodiscard]] NodeId traceback_segment(
        int64_t               seg_start,
        int64_t               seg_end,
        NodeId                end_vertex,
        const uint32_t*       bp_block,
        size_t                num_chars,
        std::vector<AlignOp>& path)
{
    int64_t row    = seg_end;
    NodeId  vertex = end_vertex;

    while (row > seg_start) {
        const size_t local = static_cast<size_t>(row - seg_start - 1);
        uint32_t enc = bp_block[local * num_chars + vertex];
        uint8_t  op  = bp_op(enc);
        NodeId   src = bp_src(enc);

        path.push_back({row, vertex, op});

        if (op == BP_INSERT) {
            // Stay at same row; follow the insertion chain one hop
            vertex = src;
        } else {
            // BP_DELETE or BP_MATCHSUB: advance to previous row
            vertex = src;
            --row;
        }
    }

    return vertex;  // vertex at row seg_start
}

// ============================================================================
// align_read_with_traceback  —  single-read entry point
//
// Workflow
//   1. Forward pass (same DP as align_read_combo) checkpointing row 0 and
//      every multiple of k to NVMe.  Peak RAM = 2 rows = 3.2 GB.
//   2. Identify best end vertex from the final row.
//   3. Free forward buffer, allocate backpointer block (k * num_chars * 4 B).
//   4. Loop backward through checkpoint segments until row 0:
//        a. Load checkpoint, recompute segment, store backpointers.
//        b. Trace backward through segment, collecting AlignOps.
//   5. Reverse path and return.
//
// Template parameters match align_combo: LogScore is the DP score type
// (int or short), Mapper is VertexLabelMapper<true/false>.
// ============================================================================

template<typename LogScore, typename Mapper>
std::vector<AlignOp> walk_traceback_from_scores(
        const LogScore*           score_all,
        int64_t                   read_len,
        size_t                    num_chars,
        NodeId                    best_vertex,
        const std::string&        read,
        const std::vector<char>&  vertex_labels,
        const InCSR&              in_char_graph,
        const InCSR&              in_str_graph,
        const Mapper&             M,
        LogScore                  match_score,
        LogScore                  subst_score,
        LogScore                  del_score,
        LogScore                  ins_score)
{
    auto vertex_of = [&](NodeId p) -> NodeId {
        if constexpr (!Mapper::hasLabels) {
            return p;
        } else {
            const auto& offs = M.vertex_label_offsets;
            auto it = std::upper_bound(offs.begin(), offs.end() - 1, p);
            return static_cast<NodeId>((it - offs.begin()) - 1);
        }
    };

    std::vector<AlignOp> path;
    path.reserve(static_cast<size_t>(read_len) +
                 static_cast<size_t>(read_len) / 4);

    NodeId  cur_v   = best_vertex;
    int64_t cur_row = read_len - 1;

    while (cur_row > 0) {
        const LogScore* row  = score_all + static_cast<size_t>(cur_row) * num_chars;
        const LogScore* prow = score_all + static_cast<size_t>(cur_row - 1) * num_chars;
        const LogScore  s    = row[cur_v];

        uint8_t op  = 255;
        NodeId  src = cur_v;

        // INSERT preferred on ties (matches compute_insert_bp overwrite)
        {
            const NodeId sv = vertex_of(cur_v);
            if constexpr (Mapper::hasLabels) {
                if (cur_v > M.first(sv) &&
                    row[cur_v - 1] + ins_score == s) {
                    op  = BP_INSERT;
                    src = static_cast<NodeId>(cur_v - 1);
                }
            }
            if (op == 255 && cur_v == M.first(sv)) {
                for (size_t ie = in_str_graph.in_offset[sv];
                     ie < in_str_graph.in_offset[sv + 1]; ++ie) {
                    const NodeId u      = in_str_graph.in_E[ie].u;
                    const NodeId u_last = M.last(u);
                    if (row[u_last] + ins_score == s) {
                        op  = BP_INSERT;
                        src = u_last;
                        break;
                    }
                }
            }
        }

        if (op == 255) {
            if (prow[cur_v] + del_score == s) {
                op  = BP_DELETE;
                src = cur_v;
            }
        }

        if (op == 255) {
            const LogScore delta = match_sub<LogScore>(
                    vertex_labels[cur_v],
                    read[static_cast<size_t>(cur_row)],
                    match_score, subst_score);
            for (size_t ie = in_char_graph.in_offset[cur_v];
                 ie < in_char_graph.in_offset[cur_v + 1]; ++ie) {
                const NodeId u = in_char_graph.in_E[ie].u;
                if (prow[u] + delta == s) {
                    op  = BP_MATCHSUB;
                    src = u;
                    break;
                }
            }
        }

        if (op == 255) {
            throw std::runtime_error(
                "score-matrix traceback: no valid predecessor at row " +
                std::to_string(cur_row) + " vertex " + std::to_string(cur_v));
        }

        path.push_back({cur_row, cur_v, op});
        cur_v = src;
        if (op != BP_INSERT)
            --cur_row;
    }

    path.push_back({0, cur_v, BP_MATCHSUB});
    std::reverse(path.begin(), path.end());
    return path;
}

template<typename LogScore, typename Mapper>
std::vector<AlignOp> align_read_with_traceback(
        const std::string&        read,
        size_t                    num_chars,
        const std::vector<char>&  vertex_labels_char,
        const InCSR&              in_char_graph,
        const Graph&              origin_str_graph,
        const sequence<NodeId>&   contracted_str_to_og,
        LogScore                  match_score,
        LogScore                  subst_score,
        LogScore                  del_score,
        LogScore                  ins_score,
        PchQuery&                 query,
        const Mapper&             Mstr,
        const CheckpointConfig&   cfg)
{
    const int64_t read_len  = static_cast<int64_t>(read.size());
    const int64_t k         = cfg.k;
    const size_t  row_bytes = num_chars * sizeof(LogScore);
    // Fast-path store: full score matrix (same bytes as bp matrix when
    // LogScore is 4B).  Score-matrix traceback skips per-cell bp writes and
    // the O(V+E) compute_insert_bp scan that dominated ConSGA's FAST tax.
    const size_t  score_store_bytes =
        static_cast<size_t>(read_len) * num_chars * sizeof(LogScore);
    const size_t  bp_store_bytes =
        static_cast<size_t>(read_len) * num_chars * sizeof(uint32_t);
#ifndef FAST_BP_MAX_BYTES_OVERRIDE
    constexpr size_t FAST_BP_MAX_BYTES = 400ULL * 1024 * 1024 * 1024;
#else
    constexpr size_t FAST_BP_MAX_BYTES = FAST_BP_MAX_BYTES_OVERRIDE;
#endif

    // Default FAST path: keep all score rows, walk by score equality.
    // -DTB_USE_BP_FAST_PATH restores the old backpointer FAST path.
#if defined(TB_USE_BP_FAST_PATH)
    const size_t fast_store_bytes = bp_store_bytes;
#else
    const size_t fast_store_bytes = score_store_bytes;
#endif

#ifdef TB_PROFILE
    g_tb_prof.reset();
    const auto _tb_read_t0 = std::chrono::steady_clock::now();
    std::fprintf(stderr,
        "TBPROF decision read_len=%lld num_chars=%zu store_GB=%.2f cap_GB=%.2f path=%s mode=%s\n",
        static_cast<long long>(read_len), num_chars,
        static_cast<double>(fast_store_bytes) / (1024.0*1024.0*1024.0),
        static_cast<double>(FAST_BP_MAX_BYTES) / (1024.0*1024.0*1024.0),
        (fast_store_bytes <= FAST_BP_MAX_BYTES) ? "FAST" : "CHECKPOINTED",
#if defined(TB_USE_BP_FAST_PATH)
        "bp"
#else
        "score"
#endif
        );
#endif

    if (fast_store_bytes <= FAST_BP_MAX_BYTES) {
#if !defined(TB_USE_BP_FAST_PATH)
        // ── Score-matrix FAST path ─────────────────────────────────────────
        UninitArray<LogScore> score_all;
        {
            TB_TIME_BLOCK(g_tb_prof.bp_alloc_s);
            const size_t ncells =
                static_cast<size_t>(read_len) * num_chars;
            score_all = UninitArray<LogScore>(ncells);
            // Parallel first-touch only for very large score matrices.
            // On mid-size graphs (e.g. Chr. 1 ~238 GB) it can slow DP via
            // NUMA placement; on Tetraodon/Opossum (~335–700 GB) it removes
            // serial page-fault cost that otherwise dominates.
            constexpr size_t kFirstTouchBytes =
                300ULL * 1024 * 1024 * 1024;
            if (ncells * sizeof(LogScore) >= kFirstTouchBytes) {
                constexpr size_t kPageScores = 4096 / sizeof(LogScore);
                const size_t npages =
                    (ncells + kPageScores - 1) / kPageScores;
                parlay::parallel_for(0, npages, [&](size_t pg) {
                    const size_t i = std::min(pg * kPageScores, ncells - 1);
                    score_all[i] = LogScore(0);
                });
            }
        }
        LogScore* prev = score_all.data();
        init_func_parallel(prev, vertex_labels_char,
                           read[0], num_chars, match_score, subst_score);

        for (int64_t i = 1; i < read_len; ++i) {
            LogScore* curr =
                score_all.data() + static_cast<size_t>(i) * num_chars;
            {
                TB_TIME_BLOCK(g_tb_prof.fwd_dp_s);
                fused_deletion_substitution<LogScore, Mapper>(
                    curr, prev,
                    vertex_labels_char, in_char_graph,
                    read[static_cast<size_t>(i)],
                    match_score, subst_score, del_score,
                    num_chars);
            }
            {
                TB_TIME_BLOCK(g_tb_prof.fwd_ins_s);
                insertion_func_parallel<LogScore, Mapper>(
                    curr, origin_str_graph, contracted_str_to_og,
                    ins_score, query, Mstr);
            }
            prev = curr;
        }

        NodeId best_vertex = 0;
        parlay::sequence<LogScore> tmins(
            parlay::num_workers(), std::numeric_limits<LogScore>::max());
        parlay::sequence<NodeId> tverts(parlay::num_workers(), NodeId(0));

        parlay::parallel_for(0, num_chars, [&](size_t v) {
            size_t tid = parlay::worker_id();
            if (prev[v] < tmins[tid]) {
                tmins[tid]  = prev[v];
                tverts[tid] = static_cast<NodeId>(v);
            }
        });

        LogScore best_score = std::numeric_limits<LogScore>::max();
        for (size_t t = 0; t < tmins.size(); ++t) {
            if (tmins[t] < best_score) {
                best_score  = tmins[t];
                best_vertex = tverts[t];
            }
        }

        InCSR in_str_graph = build_in_csr_from(origin_str_graph);
        std::vector<AlignOp> path;
        {
            TB_TIME_BLOCK(g_tb_prof.traceback_walk_s);
            path = walk_traceback_from_scores<LogScore, Mapper>(
                score_all.data(), read_len, num_chars, best_vertex,
                read, vertex_labels_char, in_char_graph, in_str_graph,
                Mstr, match_score, subst_score, del_score, ins_score);
        }
#ifdef TB_PROFILE
        {
            const double total_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - _tb_read_t0).count();
            std::fprintf(stderr,
                "TBPROF FAST(score) alloc_s=%.3f fwd_dp_s=%.3f fwd_ins_s=%.3f "
                "walk_s=%.3f total_s=%.3f\n",
                g_tb_prof.bp_alloc_s, g_tb_prof.fwd_dp_s, g_tb_prof.fwd_ins_s,
                g_tb_prof.traceback_walk_s, total_s);
        }
#endif
        return path;
#else
        // ── Legacy backpointer FAST path ───────────────────────────────────
        std::vector<LogScore> fwd(2 * num_chars);
        LogScore* prev = fwd.data();
        LogScore* curr = fwd.data() + num_chars;
        UninitU32Array bp_all;
        {
            TB_TIME_BLOCK(g_tb_prof.bp_alloc_s);
            bp_all = UninitU32Array(static_cast<size_t>(read_len) * num_chars);
        }

        init_func_parallel(prev, vertex_labels_char,
                           read[0], num_chars, match_score, subst_score);

        for (int64_t i = 1; i < read_len; ++i) {
            uint32_t* bp_row =
                bp_all.data() + static_cast<size_t>(i) * num_chars;

            {
                TB_TIME_BLOCK(g_tb_prof.fwd_dp_s);
                fused_del_sub_with_bp<LogScore>(
                    curr, bp_row, prev,
                    vertex_labels_char, in_char_graph,
                    read[static_cast<size_t>(i)],
                    match_score, subst_score, del_score,
                    num_chars);
            }

            {
                TB_TIME_BLOCK(g_tb_prof.fwd_ins_s);
                insertion_func_parallel<LogScore, Mapper>(
                    curr, origin_str_graph, contracted_str_to_og,
                    ins_score, query, Mstr);
            }

            {
                TB_TIME_BLOCK(g_tb_prof.fwd_insbp_s);
                compute_insert_bp<LogScore, Mapper>(
                    curr, bp_row, origin_str_graph, Mstr, ins_score);
            }

            std::swap(prev, curr);
        }

        NodeId best_vertex = 0;
        parlay::sequence<LogScore> tmins(
            parlay::num_workers(), std::numeric_limits<LogScore>::max());
        parlay::sequence<NodeId> tverts(parlay::num_workers(), NodeId(0));

        parlay::parallel_for(0, num_chars, [&](size_t v) {
            size_t tid = parlay::worker_id();
            if (prev[v] < tmins[tid]) {
                tmins[tid]  = prev[v];
                tverts[tid] = static_cast<NodeId>(v);
            }
        });

        LogScore best_score = std::numeric_limits<LogScore>::max();
        for (size_t t = 0; t < tmins.size(); ++t) {
            if (tmins[t] < best_score) {
                best_score  = tmins[t];
                best_vertex = tverts[t];
            }
        }

        std::vector<AlignOp> path;
        path.reserve(static_cast<size_t>(read_len) +
                     static_cast<size_t>(read_len) / 4);

        {
            TB_TIME_BLOCK(g_tb_prof.traceback_walk_s);
            NodeId  cur_vertex = best_vertex;
            int64_t cur_row    = read_len - 1;
            while (cur_row > 0) {
                const uint32_t enc =
                    bp_all[static_cast<size_t>(cur_row) * num_chars
                           + static_cast<size_t>(cur_vertex)];
                const uint8_t op  = bp_op(enc);
                const NodeId  src = bp_src(enc);

                path.push_back({cur_row, cur_vertex, op});

                if (op == BP_INSERT) {
                    cur_vertex = src;
                } else {
                    cur_vertex = src;
                    --cur_row;
                }
            }

            path.push_back({0, cur_vertex, BP_MATCHSUB});
            std::reverse(path.begin(), path.end());
        }
#ifdef TB_PROFILE
        {
            const double total_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - _tb_read_t0).count();
            std::fprintf(stderr,
                "TBPROF FAST bp_alloc_s=%.3f fwd_dp_s=%.3f fwd_ins_s=%.3f "
                "fwd_insbp_s=%.3f walk_s=%.3f total_s=%.3f\n",
                g_tb_prof.bp_alloc_s, g_tb_prof.fwd_dp_s, g_tb_prof.fwd_ins_s,
                g_tb_prof.fwd_insbp_s, g_tb_prof.traceback_walk_s, total_s);
        }
#endif
        return path;
#endif // TB_USE_BP_FAST_PATH
    }

    CheckpointManager ckpt(cfg, row_bytes);

    // ── Phase 1: forward pass with checkpointing ─────────────────────────────
    //
    // We limit peak RAM to 2 rows by immediately saving to NVMe and swapping
    // the two-row buffer.  The forward buffer is freed before the large
    // backpointer block is allocated in Phase 3.

    NodeId best_vertex = 0;

    {   // inner scope: forward buffer lives only here
        std::vector<LogScore> fwd(2 * num_chars);
        LogScore* prev = fwd.data();
        LogScore* curr = fwd.data() + num_chars;

        // Row 0: initialise from the first read base
        init_func_parallel(prev, vertex_labels_char,
                           read[0], num_chars, match_score, subst_score);
        {
            TB_TIME_BLOCK(g_tb_prof.ckpt_save_s);
            ckpt.save(0, prev);
        }

        for (int64_t i = 1; i < read_len; ++i) {
            {
                TB_TIME_BLOCK(g_tb_prof.fwd_dp_s);
                fused_deletion_substitution<LogScore, Mapper>(
                    curr, prev,
                    vertex_labels_char, in_char_graph,
                    read[static_cast<size_t>(i)],
                    match_score, subst_score, del_score,
                    num_chars);
            }

            {
                TB_TIME_BLOCK(g_tb_prof.fwd_ins_s);
                insertion_func_parallel<LogScore, Mapper>(
                    curr, origin_str_graph, contracted_str_to_og,
                    ins_score, query, Mstr);
            }

            std::swap(prev, curr);

            if (i % k == 0) {
                TB_TIME_BLOCK(g_tb_prof.ckpt_save_s);
                ckpt.save(i, prev);
            }
        }

        // ── Phase 2: find best end vertex ────────────────────────────────────
        //
        // prev now holds row (read_len - 1).  Parallel reduction over all
        // num_chars columns.

        parlay::sequence<LogScore> tmins(
            parlay::num_workers(), std::numeric_limits<LogScore>::max());
        parlay::sequence<NodeId> tverts(parlay::num_workers(), NodeId(0));

        parlay::parallel_for(0, num_chars, [&](size_t v) {
            size_t tid = parlay::worker_id();
            if (prev[v] < tmins[tid]) {
                tmins[tid]  = prev[v];
                tverts[tid] = static_cast<NodeId>(v);
            }
        });

        LogScore best_score = std::numeric_limits<LogScore>::max();
        for (size_t t = 0; t < tmins.size(); ++t) {
            if (tmins[t] < best_score) {
                best_score  = tmins[t];
                best_vertex = tverts[t];
            }
        }
        // fwd goes out of scope → ~3.2 GB freed before bp_block below
    }

    // ── Phase 3: backward traceback ──────────────────────────────────────────
    //
    // Allocate one backpointer block large enough for the widest segment we
    // will ever recompute.  No segment spans more than k rows by
    // construction, and none spans more than read_len rows either (there are
    // only read_len rows in total), so min(k, read_len) is a safe, tight
    // upper bound.  For short reads on huge graphs (e.g. 250 bp on Opossum
    // with k=500) this halves the allocation versus always sizing for k.
    // We reuse this block for every segment, keeping peak RAM here at
    // min(k, read_len) × num_chars × 4 bytes.  UninitU32Array skips the
    // zero-initialization std::vector would otherwise perform: every cell is
    // overwritten by fused_del_sub_with_bp/compute_insert_bp before it is
    // read, so the zero-fill was pure (and, at this scale, dominant) waste.

    const int64_t  bp_block_rows = std::min<int64_t>(k, read_len);
    UninitU32Array bp_block(static_cast<size_t>(bp_block_rows) * num_chars);
    std::vector<LogScore>  ckpt_row(num_chars);
    std::vector<AlignOp>   path;
    path.reserve(static_cast<size_t>(read_len) +
                 static_cast<size_t>(read_len) / 4); // slight overalloc

    NodeId  cur_vertex = best_vertex;
    int64_t cur_row    = read_len - 1;

    while (cur_row > 0) {
        // Determine the segment [seg_start+1 … seg_end].
        //
        // When cur_row is exactly on a checkpoint boundary (multiple of k) we
        // must step back one full segment to cover row cur_row itself; the
        // checkpoint at cur_row was saved during the forward pass and will
        // serve as the END of the segment, not the start.
        int64_t seg_start = (cur_row % k == 0)
                            ? (cur_row - k)
                            : (cur_row / k * k);
        if (seg_start < 0) seg_start = 0;
        int64_t seg_end = cur_row;

        // Load the checkpoint score row at seg_start
        {
            TB_TIME_BLOCK(g_tb_prof.ckpt_load_s);
            ckpt.load(seg_start, ckpt_row.data());
        }

        // Recompute segment and store backpointers
        recompute_segment_with_bp<LogScore, Mapper>(
            read,
            seg_start, seg_end,
            ckpt_row.data(),
            bp_block.data(),
            num_chars,
            vertex_labels_char,
            in_char_graph,
            origin_str_graph,
            contracted_str_to_og,
            match_score, subst_score, del_score, ins_score,
            query, Mstr);

        // Trace backward through the segment, appending reversed AlignOps
        NodeId landing;
        {
            TB_TIME_BLOCK(g_tb_prof.traceback_walk_s);
            landing = traceback_segment(
                seg_start, seg_end,
                cur_vertex,
                bp_block.data(),
                num_chars,
                path);
        }

#ifdef TB_PROFILE
        ++g_tb_prof.n_segments;
#endif

        // Free the checkpoint file now that we are done with it (optional)
        if (cfg.cleanup && seg_start > 0) {
            TB_TIME_BLOCK(g_tb_prof.ckpt_erase_s);
            ckpt.erase(seg_start);
        }

        cur_vertex = landing;
        cur_row    = seg_start;
    }

    // Record row 0: the alignment starts here.
    // Row 0 has no predecessor row; the init DP is a match/sub assignment,
    // so BP_MATCHSUB is the natural label.
    path.push_back({0, cur_vertex, BP_MATCHSUB});
    if (cfg.cleanup) {
        TB_TIME_BLOCK(g_tb_prof.ckpt_erase_s);
        ckpt.erase(0);
    }

    // Path was built in reverse; flip to forward order
    std::reverse(path.begin(), path.end());

#ifdef TB_PROFILE
    {
        const double total_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - _tb_read_t0).count();
        std::fprintf(stderr,
            "TBPROF CHECKPOINTED fwd_dp_s=%.3f fwd_ins_s=%.3f ckpt_save_s=%.3f "
            "ckpt_load_s=%.3f seg_dp_s=%.3f seg_ins_s=%.3f seg_insbp_s=%.3f "
            "traceback_walk_s=%.3f ckpt_erase_s=%.3f n_segments=%ld total_s=%.3f\n",
            g_tb_prof.fwd_dp_s, g_tb_prof.fwd_ins_s, g_tb_prof.ckpt_save_s,
            g_tb_prof.ckpt_load_s, g_tb_prof.seg_dp_s, g_tb_prof.seg_ins_s,
            g_tb_prof.seg_insbp_s, g_tb_prof.traceback_walk_s,
            g_tb_prof.ckpt_erase_s, g_tb_prof.n_segments, total_s);
    }
#endif
    return path;
}

// ============================================================================
// align_combo_with_traceback  —  batch entry point
//
// Mirrors the align_combo interface; processes reads sequentially, each
// using a separate subdirectory for its checkpoint files.
// ============================================================================

template<typename LogScore, typename Mapper>
std::vector<std::vector<AlignOp>> align_combo_with_traceback(
        const std::vector<std::string>&  reads,
        const std::vector<char>&         vertex_labels_char,
        const InCSR&                     in_char_graph,
        const Graph&                     origin_str_graph,
        const sequence<NodeId>&          contracted_str_to_og,
        LogScore                         match_score,
        LogScore                         subst_score,
        LogScore                         del_score,
        LogScore                         ins_score,
        PchQuery&                        query,
        const Mapper&                    Mstr,
        const CheckpointConfig&          cfg)
{
    const size_t num_chars = vertex_labels_char.size();
    std::vector<std::vector<AlignOp>> results;
    results.reserve(reads.size());

    for (size_t ri = 0; ri < reads.size(); ++ri) {
        CheckpointConfig rcfg = cfg;
        rcfg.dir = cfg.dir + "/read_" + std::to_string(ri);

        results.push_back(
            align_read_with_traceback<LogScore, Mapper>(
                reads[ri], num_chars,
                vertex_labels_char, in_char_graph,
                origin_str_graph, contracted_str_to_og,
                match_score, subst_score, del_score, ins_score,
                query, Mstr, rcfg));
    }
    return results;
}

// ============================================================================
// ops_to_cigar  —  convert alignment path to a CIGAR string
//
// Convention: M = match or mismatch, D = deletion (gap in reference / graph),
//             I = insertion (gap in read).  Run-length encoded.
// ============================================================================

[[nodiscard]] inline std::string ops_to_cigar(const std::vector<AlignOp>& path) {
    std::string cigar;
    if (path.empty()) return cigar;

    auto op_char = [](uint8_t op) -> char {
        switch (op) {
            case BP_DELETE:   return 'D';
            case BP_MATCHSUB: return 'M';
            default:          return 'I';
        }
    };

    char prev_c = op_char(path[0].op);
    int  run    = 1;

    for (size_t i = 1; i < path.size(); ++i) {
        char c = op_char(path[i].op);
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

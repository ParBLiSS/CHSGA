#include "query.hpp"
#include "csr_align.hpp"
#ifdef USE_PAPI
#include <papi.h>

// Events to track (AMD EPYC native perf_event names)
static constexpr const char* PAPI_EVENTS[] = {
  "perf::CPU-CYCLES",              // cycles
  "perf::INSTRUCTIONS",            // retired instructions
  "perf::BRANCH-MISSES",           // branch mispredicts
  "perf::L1-DCACHE-LOAD-MISSES",   // L1 data cache load misses
  "perf::L1-DCACHE-LOADS"          // total L1 data cache loads
};
static constexpr int NUM_PAPI_EVENTS = sizeof(PAPI_EVENTS)/sizeof(PAPI_EVENTS[0]);

// Initialize once
inline void papi_init_once() {
  static bool inited = false;
  if (inited) return;
  int rv = PAPI_library_init(PAPI_VER_CURRENT);
  if (rv != PAPI_VER_CURRENT) {
    std::fprintf(stderr, "[PAPI] init failed: %d\n", rv);
    std::exit(1);
  }
  inited = true;
}

// Aggregator: totals + call counts
struct PapiAggregator {
  struct Bucket {
    std::array<long long, NUM_PAPI_EVENTS> totals{};
    long long calls = 0;
  };
  std::map<std::string, Bucket> buckets;

  void add(const std::string& name,
           const std::array<long long, NUM_PAPI_EVENTS>& vals) {
    auto& b = buckets[name];
    for (int i = 0; i < NUM_PAPI_EVENTS; ++i) b.totals[i] += vals[i];
    b.calls++;
  }

    void report() {
// === PAPI Summary ===
// For each measured phase (e.g. deletion_func_parallel, substitution_func_parallel),
// we print both raw totals across all calls, averages per call, and derived metrics.
//
// Raw totals:
//   cycles          = total CPU cycles consumed in this phase
//   inst            = total retired instructions
//   br_misses       = total branch mispredictions
//   l1_load_misses  = total L1 data cache load misses
//   l1_loads        = total L1 data cache load operations
//
// Averages per call:
//   cycles          = average cycles per invocation of this phase
//   inst            = average instructions retired per invocation
//   br_misses       = average branch mispredicts per invocation
//   l1_load_misses  = average L1 load misses per invocation
//   l1_loads        = average L1 load operations per invocation
//
// Derived metrics (normalized, easier to interpret):
//   IPC             = instructions per cycle (inst / cycles).
//                     Higher IPC means better utilization of the core.
//   Branch MPKI     = branch mispredicts per 1000 instructions.
//                     High MPKI indicates branch-heavy code with poor prediction.
//   L1 MPKI         = L1 load misses per 1000 instructions.
//                     High MPKI means frequent cache misses relative to instruction count.
//   L1 miss rate    = fraction of L1 load operations that missed (misses / loads).
//                     Expressed as a percentage; higher values mean poor cache locality.
//
// These derived metrics help distinguish whether a kernel is compute-bound,
// branch-bound, or memory-bound, beyond just raw cycle counts.

    std::fprintf(stderr, "\n=== PAPI Summary ===\n");
    for (auto& kv : buckets) {
        const auto& name = kv.first;
        const auto& b = kv.second;
        const auto& v = b.totals;

        double avg_cycles   = b.calls ? (double)v[0]/b.calls : 0.0;
        double avg_inst     = b.calls ? (double)v[1]/b.calls : 0.0;
        double avg_br_miss  = b.calls ? (double)v[2]/b.calls : 0.0;
        double avg_l1_miss  = b.calls ? (double)v[3]/b.calls : 0.0;
        double avg_l1_loads = b.calls ? (double)v[4]/b.calls : 0.0;

        // Derived metrics
        double ipc          = v[0] ? (double)v[1]/v[0] : 0.0;
        double branch_mpki  = v[1] ? (double)v[2]/(v[1]/1000.0) : 0.0;
        double l1_mpki      = v[1] ? (double)v[3]/(v[1]/1000.0) : 0.0;
        double l1_miss_rate = v[4] ? (double)v[3]/v[4] : 0.0;

        std::fprintf(stderr,
        "[%s] calls=%lld\n"
        "   totals: cycles=%lld inst=%lld br_misses=%lld l1_load_misses=%lld l1_loads=%lld\n"
        "   avg/call: cycles=%.1f inst=%.1f br_misses=%.1f l1_load_misses=%.1f l1_loads=%.1f\n"
        "   derived: IPC=%.2f Branch MPKI=%.2f L1 MPKI=%.2f L1 miss rate=%.2f%%\n",
        name.c_str(), b.calls,
        v[0], v[1], v[2], v[3], v[4],
        avg_cycles, avg_inst, avg_br_miss, avg_l1_miss, avg_l1_loads,
        ipc, branch_mpki, l1_mpki, l1_miss_rate*100.0
        );
    }
    std::fprintf(stderr, "====================\n");
    }
};
static PapiAggregator papiAgg;

// Phase RAII object
struct PapiPhase {
  std::string name;
  int event_set = PAPI_NULL;
  std::array<long long, NUM_PAPI_EVENTS> values{};

  explicit PapiPhase(const char* phase_name) : name(phase_name) {
    papi_init_once();
    PAPI_create_eventset(&event_set);
    for (int i = 0; i < NUM_PAPI_EVENTS; ++i) {
      // add native events by name
      PAPI_add_named_event(event_set, PAPI_EVENTS[i]);
    }
    PAPI_start(event_set);
  }
  ~PapiPhase() {
    PAPI_stop(event_set, values.data());
    PAPI_cleanup_eventset(event_set);
    PAPI_destroy_eventset(&event_set);
    papiAgg.add(name, values); // aggregate instead of printing
  }
};
#endif

double time_diff_sec(const timespec &start, const timespec &end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

constexpr size_t MAX_DEBUG_PRINTABLE_VERTICES = 20;


void print_origin_graph(const Graph &origin_graph) {
    if(origin_graph.n <= MAX_DEBUG_PRINTABLE_VERTICES) {
        for(size_t u = 0; u < origin_graph.n; u++) {
            std::cout<<origin_graph.offset[u]<<" ";
            for(size_t e = origin_graph.offset[u]; e < origin_graph.offset[u+1]; e++) {
                std::cout<<"("<<u<<","<<origin_graph.E[e].v<<") ";
            }
        }
        std::cout<<std::endl;
    }
}

void print_contracted_graph(const PchGraph &contracted_graph, const sequence<NodeId> &contracted_to_og_mapping) {
    if(contracted_graph.n <= MAX_DEBUG_PRINTABLE_VERTICES) {
        for(size_t i = 0; i < contracted_graph.n; i++) {
            std::cout<<"("<<contracted_to_og_mapping[i]<<","<<contracted_graph.level[i]<<") ";
        }
        std::cout<<std::endl;
        
        for(size_t u = 0; u < contracted_graph.n; u++) {
            std::cout<<contracted_graph.offset[u]<<" ";
            for(size_t e = contracted_graph.offset[u]; e < contracted_graph.offset[u+1]; e++) {
                std::cout<<"("<<contracted_to_og_mapping[u]<<","<<contracted_to_og_mapping[contracted_graph.E[e].v]<<","<<contracted_graph.E[e].w<<") ";
            }
        }
        std::cout<<std::endl;
        
        for(size_t u = 0; u < contracted_graph.n; u++) {
            std::cout<<contracted_graph.in_offset[u]<<" ";
            for(size_t e = contracted_graph.in_offset[u]; e < contracted_graph.in_offset[u+1]; e++) {
                std::cout<<"("<<contracted_to_og_mapping[u]<<","<<contracted_to_og_mapping[contracted_graph.in_E[e].v]<<","<<contracted_graph.in_E[e].w<<") ";
            }
        }
        std::cout<<std::endl;
    }
}

template<bool UseVertexLabels>
struct VertexLabelMapper {
    static constexpr bool hasLabels = UseVertexLabels;
    const std::vector<size_t>& vertex_label_offsets;

  VertexLabelMapper(const std::vector<size_t>& offs)
    : vertex_label_offsets(offs) {}

  // first label if labels enabled, else just c
  inline NodeId first(NodeId c) const {
    if constexpr(UseVertexLabels) {
      return vertex_label_offsets[c];
    } else {
      return static_cast<NodeId>(c);
    }
  }

  // last  label if labels enabled, else just c
  inline NodeId last(NodeId c) const {
    if constexpr(UseVertexLabels) {
      return vertex_label_offsets[c+1] - 1;
    } else {
      return static_cast<NodeId>(c);
    }
  }
};


template<typename LogScore>
void init_func_parallel(LogScore* const first_row, const std::vector<char>& vertex_labels, char const first_base, size_t const num_chars, const LogScore match_score, const LogScore subst_score) {
    parallel_for(0, num_chars, [&](NodeId v) {
        first_row[v] = match_sub<LogScore>(vertex_labels[v], first_base, match_score, subst_score);
    });
}



struct InCSR {
  std::vector<size_t> in_offset; // size G.n + 1
  struct Edge { NodeId u; };
  std::vector<Edge> in_E;
};

inline InCSR build_in_csr(Graph&& G) {
  InCSR in;
  in.in_offset.assign(G.n + 1, 0);

  // Count in-degree
  for (NodeId u = 0; u < G.n; ++u)
    for (EdgeId e = G.offset[u]; e < G.offset[u+1]; ++e)
      ++in.in_offset[G.E[e].v + 1];

  // Prefix sum
  for (NodeId v = 0; v < G.n; ++v)
    in.in_offset[v + 1] += in.in_offset[v];

  // Fill incoming edges
  in.in_E.resize(in.in_offset[G.n]);
  std::vector<size_t> cur = in.in_offset;
  for (NodeId u = 0; u < G.n; ++u)
    for (EdgeId e = G.offset[u]; e < G.offset[u+1]; ++e) {
      NodeId v = G.E[e].v;
      in.in_E[cur[v]++] = InCSR::Edge{u};
    }
  return in;
}


template<typename LogScore, typename Mapper>
void fused_deletion_substitution(LogScore*       edits_dst,
                                 const LogScore* edits_src,
                                 const std::vector<char>& vertex_labels,
                                 const InCSR&    inG,
                                 char            read_base,
                                 LogScore        match_score,
                                 LogScore        subst_score,
                                 LogScore        del_score,
                                 size_t          num_chars) {
  #ifdef USE_PAPI
    PapiPhase _pp("fused_deletion_substitution");
  #endif
  const size_t block_size = 4096; // tune this (256..4096) for your workload / cache
  parlay::blocked_for(0, num_chars, block_size,
    [&](size_t /*block_no*/, size_t start, size_t end) {
      for (size_t vpos = start; vpos < end; ++vpos) {
        // Start with deletion candidate
        LogScore best = edits_src[vpos] + del_score;
        LogScore delta = match_sub<LogScore>(vertex_labels[vpos], read_base, match_score, subst_score);
        // Consider all incoming edges (substitution/match)
        for (size_t ie = inG.in_offset[vpos]; ie < inG.in_offset[vpos + 1]; ++ie) {
          NodeId u     = inG.in_E[ie].u;
          LogScore src = edits_src[u];
          LogScore cand  = src + delta;
          if (cand < best) best = cand;
        }

        edits_dst[vpos] = best;
      }
    }
  );
}

template <typename LogScore, typename Mapper>
void insertionQueryLayerParallelization(
    const PchQuery& query,
    LogScore ins_score,
    LogScore* const edit_scores,
    const sequence<NodeId>& contracted_to_og,
    const Mapper& M
) {
  // Delegate all index mapping to Mapper M
  auto idx_u_fwd = [&](NodeId c) { return M.last(c); };
  auto idx_v_fwd = [&](NodeId c) { return M.first(c); };
  auto idx_u_bwd = [&](NodeId c) { return M.first(c); };
  auto idx_v_bwd = [&](NodeId c) { return M.last(c); };

  // Loop over each connected component
  for (NodeId cc = 0; cc + 1 < query.GC.ccOffset.size(); ++cc) {
    // —— FORWARD: ascend layers low → high ——
    for (size_t layer = query.GC.ccOffset[cc]; layer < query.GC.ccOffset[cc+1]; ++layer) {
      parallel_for(
        query.GC.layerOffset[layer],
        query.GC.layerOffset[layer+1],
        [&](NodeId u) {
          NodeId c_u  = contracted_to_og[u];
          size_t u_ix = idx_u_fwd(c_u);
            for (size_t j = query.GC.offset[u]; j < query.GC.offset[u+1]; ++j) {
              NodeId c_v  = contracted_to_og[query.GC.E[j].v];
              size_t v_ix = idx_v_fwd(c_v);
              EdgeTy w    = query.GC.E[j].w;

              LogScore prev = edit_scores[v_ix];
              LogScore next = edit_scores[u_ix] + w * ins_score;
              cas_update<LogScore>(&edit_scores[v_ix], prev, next);
            }     
        }
      );
    }

    // —— BACKWARD: descend layers high → low ——
    for (size_t layer = query.GC.ccOffset[cc+1]; layer-- > query.GC.ccOffset[cc]; ) {
      parallel_for(
        query.GC.layerOffset[layer],
        query.GC.layerOffset[layer+1],
        [&](NodeId u) {
          NodeId c_u  = contracted_to_og[u];
          size_t u_ix = idx_u_bwd(c_u);
          LogScore &target_score = edit_scores[u_ix];

          const size_t in_begin = query.GC.in_offset[u];
          const size_t in_end   = query.GC.in_offset[u+1];

          for (size_t j = in_begin; j < in_end; ++j) {
            NodeId orig_v = query.GC.in_E[j].v;
            NodeId c_v    = contracted_to_og[orig_v];
            size_t v_ix   = idx_v_bwd(c_v);
            EdgeTy w      = query.GC.in_E[j].w;

            LogScore candidate = edit_scores[v_ix] + w * ins_score;

            if (candidate < target_score) {
              target_score = candidate;
            }
          }
        }
      );
    }
  }
}

template<typename LogScore, typename Mapper>
void insertion_func_parallel(
    LogScore* const                    edits_dst,
    const Graph&                       origin_graph,
    const sequence<NodeId>&            contracted_to_og_mapping,
    const LogScore                     ins_score,
    const PchQuery&                    query,
    const Mapper&                      M
) {
    #ifdef USE_PAPI
    PapiPhase _pp("insertion_func_parallel");
    #endif
    //
    // 1) initial “inside-node” insertions (only if labels enabled)
    //
    if constexpr (Mapper::hasLabels) {
        parallel_for(0, origin_graph.n, [&](NodeId u) {
            size_t start = M.first(u);
            size_t end   = M.last(u);
            for (size_t v = start; v < end; ++v) {
                auto cand = edits_dst[v] + ins_score;
                if (cand < edits_dst[v+1]) {
                    edits_dst[v+1] = cand;
                }
            }
        });
    }
    insertionQueryLayerParallelization(
        query, ins_score, edits_dst, contracted_to_og_mapping, M
    );

    //
    // 3) final “inside-node” propagation (only if labels enabled)
    //
    if constexpr (Mapper::hasLabels) {
        parallel_for(0, origin_graph.n, [&](NodeId u) {
            size_t start = M.first(u);
            size_t end   = M.last(u);
            size_t v     = start;
            while (v < end) {
                auto cand = edits_dst[v] + ins_score;
                if (cand < edits_dst[v+1]) {
                    edits_dst[v+1] = cand;
                    ++v;
                } else {
                    break;
                }
            }
        });
    }
}

template<typename LogScore, typename MapperChar, typename MapperStr>
LogScore* align_read_combo(const std::string &read,
                           LogScore* edit_array,
                           size_t num_chars,
                           const std::vector<char> &vertex_labels_char,
                           const InCSR &in_char_graph,
                           const Graph &origin_str_graph,
                           const sequence<NodeId> &contracted_str_to_og,
                           LogScore match_score,
                           LogScore subst_score,
                           LogScore del_score,
                           LogScore ins_score,
                           PchQuery &query,
                           double &total_time,
                           double &substitution_time,
                           double &insertion_time,
                           const MapperStr& Mstr) {
    timespec total_start, total_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    // Inline initialization for the first character of the read.
    // Set the 0th row (edits_src) to match/subst costs for aligning read[0] to each vertex.
    init_func_parallel(edit_array, vertex_labels_char, read[0], num_chars, match_score, subst_score);

    auto readLength = read.length();
    LogScore* edits_src = edit_array;
    LogScore* edits_dst = edit_array + num_chars;

    substitution_time = 0.0;
    insertion_time = 0.0;

    for (size_t i = 1; i < readLength; i++) {
        char read_base = read[i];

        // Substitutions: call the provided substitution_func_parallel using character graphs and Mchar
        timespec sub_start, sub_end;
        clock_gettime(CLOCK_MONOTONIC, &sub_start);
        fused_deletion_substitution<LogScore, MapperChar>(
                edits_dst, edits_src,
                vertex_labels_char,
                in_char_graph,
                read_base,
                match_score, subst_score,
                del_score,
                num_chars
            );
        clock_gettime(CLOCK_MONOTONIC, &sub_end);
        substitution_time += time_diff_sec(sub_start, sub_end);

        // Insertions: call the provided insertion_func_parallel using string graphs and Mstr
        timespec ins_start, ins_end;
        clock_gettime(CLOCK_MONOTONIC, &ins_start);
        insertion_func_parallel<LogScore, MapperStr>(
            edits_dst,
            origin_str_graph,
            contracted_str_to_og,
            ins_score,
            query,
            Mstr
        );
        clock_gettime(CLOCK_MONOTONIC, &ins_end);
        insertion_time += time_diff_sec(ins_start, ins_end);

        std::swap(edits_src, edits_dst);
    }

    clock_gettime(CLOCK_MONOTONIC, &total_end);
    total_time = time_diff_sec(total_start, total_end);
    return edits_src; // edits_src is result row because of pointer swap in last iteration
}

template<typename LogScore, typename MapperStr>
std::vector<LogScore> align_combo(const std::vector<std::string> &reads,
                                  const std::vector<char> &vertex_labels_char,
                                  const InCSR &in_char_graph,
                                  const Graph &origin_str_graph,
                                  const sequence<NodeId> &contracted_str_to_og,
                                  const LogScore match_score,
                                  const LogScore subst_score,
                                  const LogScore del_score,
                                  const LogScore ins_score,
                                  PchQuery &query,
                                  const MapperStr& Mstr) {
    size_t const num_chars = vertex_labels_char.size();
    LogScore* edit_array = new LogScore[2 * num_chars];
    std::vector<LogScore> min_scores;
    parlay::sequence<LogScore> local_mins(parlay::num_workers());

    std::vector<double> total_times, substitution_times, insertion_times;
    for (const std::string &read : reads) {
        double total_time = 0.0, substitution_time = 0.0, insertion_time = 0.0;

        LogScore* result_row = align_read_combo<LogScore, MapperStr>(
            read,
            edit_array,
            num_chars,
            vertex_labels_char,
            in_char_graph,
            origin_str_graph,
            contracted_str_to_og,
            match_score, subst_score, del_score, ins_score,
            query,
            total_time, substitution_time, insertion_time,
            Mstr
        );

        std::fill(local_mins.begin(), local_mins.end(), std::numeric_limits<LogScore>::max());
        parallel_for(0, num_chars, [&](size_t v) {
            size_t tid = parlay::worker_id();
            LogScore val = result_row[v];
            if (val < local_mins[tid]) {
                local_mins[tid] = val;
            }
        });
        LogScore min_score = parlay::reduce(local_mins, parlay::minm<LogScore>());

        total_times.push_back(total_time);
        substitution_times.push_back(substitution_time);
        insertion_times.push_back(insertion_time);
        min_scores.push_back(min_score);
    }

    std::cout << "Average times (skipping first read), 95% CI:\n";
    print_mean_with_moe(" Total", total_times);
    print_mean_with_moe(" Substitutions", substitution_times);
    print_mean_with_moe(" Insertions", insertion_times);

    delete [] edit_array;
    #ifdef USE_PAPI
    papiAgg.report();
    #endif
    return min_scores;
}

#include <algorithm>
#include <vector>
#include <string>
#include <limits.h>
#include <omp.h>
#include <numeric> // for reduce
#include <cmath> // for sqrt
#include "graph_util.hpp"
#include <deque>
#include <unordered_map>

template <typename T>
void printVector(const std::string& name, const std::vector<T>& vec) {
    std::cout << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << static_cast<int64_t>(vec[i]);
        if (i < vec.size() - 1) {
            std::cout << " ";
        }
    }
        std::cout << '\n';
}

enum Operation {
    TOTAL = 0,
    DELETION = 1,
    INSERTION = 2,
    MATCHMISMATCH = 3,
    NUM_OPERATIONS 
};
std::string operationToString(Operation op) {
    switch (op) {
        case TOTAL: return "Total";
        case DELETION: return "Deletions";
        case INSERTION: return "Insertions";
        case MATCHMISMATCH: return "Substitutions";
        default: return "UNKNOWN_OPERATION";
    }
}

template<typename LogScore>
inline LogScore match_sub(const char &a, const char &b, const LogScore &match_score, const LogScore &subst_score) {
  #if ALPHABET_SIZE == 5
  return (a==b) ? match_score : subst_score;
  #elif ALPHABET_SIZE == 4
  return (a=='N'||b=='N'||a==b) ? match_score : subst_score;
  #else
  #error "ALPHABET_SIZE must be 4 or 5"
  #endif
}

double mean_skip_first(const std::vector<double>& v) {
    if (v.size() <= 1) return 0.0;
    size_t n = v.size() - 1;
    double sum = std::accumulate(v.begin() + 1, v.end(), 0.0);
    return sum / static_cast<double>(n);
}

double sample_sd_skip_first(const std::vector<double>& v) {
    if (v.size() <= 2) return 0.0;
    size_t n = v.size() - 1;
    double m = mean_skip_first(v);
    double accum = 0.0;
    for (size_t i = 1; i < v.size(); ++i) {
        double d = v[i] - m;
        accum += d * d;
    }
    return std::sqrt(accum / static_cast<double>(n - 1));
}

double t_critical_95(int df) {
    // Two-sided 95% t critical values for df = 1..19 (approx)
    static const std::unordered_map<int,double> t95 = {
        {1,12.706}, {2,4.303}, {3,3.182}, {4,2.776}, {5,2.571},
        {6,2.447}, {7,2.365}, {8,2.306}, {9,2.262}, {10,2.228},
        {11,2.201}, {12,2.179}, {13,2.160}, {14,2.145}, {15,2.131},
        {16,2.120}, {17,2.110}, {18,2.101}, {19,2.093}, {20,2.086}
    };
    if (df <= 0) return 0.0;
    if (df >= 20) return t95.at(20);
    return t95.at(df);
}

void print_mean_with_moe(const std::string& label, const std::vector<double>& v) {
    if (v.size() <= 1) {
        std::cout << label << ": insufficient data (need >1 after skipping first)\n";
        return;
    }
    size_t n = v.size() - 1;
    double m = mean_skip_first(v);
    double s = sample_sd_skip_first(v);
    int df = static_cast<int>(n) - 1;
    double tcrit = t_critical_95(df);
    double se = (n > 0) ? s / std::sqrt(static_cast<double>(n)) : 0.0;
    double moe = tcrit * se;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << label << ": " << m << " ± " << moe << " sec (n=" << n << ", df=" << df << ")\n";
}

void serial_timing_calculations(double time, const int num_reads_so_far, double &running_time, double &running_time_var) {
  double old_time_mean = running_time;
  running_time += (time - old_time_mean) / num_reads_so_far;
  running_time_var += (time-running_time)*(time-old_time_mean);
}

template<typename LogScore>
std::vector<LogScore> serial_align_CSR_correctness(const std::vector<std::string> &reads, const int &numVertices, const int numEdges, const std::vector<char> &vertex_labels, const std::vector<int> &offsets_out, const std::vector<int> &edges, const LogScore &match_score, const LogScore &subst_score, const LogScore &del_score, const LogScore &ins_score, const bool &match_cut, double (&running_time)[NUM_OPERATIONS], double (&stddev_time)[NUM_OPERATIONS]) {
  std::cout << "assuming all reads have same length" << std::endl;
  auto readLength = reads[0].length();
  LogScore* edit_array = new LogScore[readLength * numVertices];
  //LogScore* edits[2];
  
  LogScore* edits_dst;
  LogScore* edits_src;
  //double running_time[NUM_OPERATIONS] = {0.0};
  double running_time_var[NUM_OPERATIONS] = {0.0};
  std::vector<LogScore> min_scores;
  int num_reads_so_far = 0;
  for (const std::string &read : reads) {
    //edits[0] = edit_array;
    //edits[1] = edit_array + numVertices;
    edits_dst = edit_array;
    for(int32_t i = 0; i < numVertices; i++) {
      //edits[1][i] = 0; // base case of 0 if 1-indexed can become -1 for 0-indexed: -1 & 1 = 1
      edits_dst[i] = match_sub(read[0],vertex_labels[i],match_score,subst_score); // im changing to starting with base case at read position 1
    }
    double times[NUM_OPERATIONS]  = {0.0};
    double start_times[NUM_OPERATIONS];
    auto readLength = read.length();
    //for(int32_t i = 0; i < readLength; i++) {
    for(int32_t i = 1; i < readLength; i++) {
      char read_base = read[i];
      //bool const src = (i - 1) & 1, dst = i & 1;
      edits_src = edits_dst;
      edits_dst = edits_dst + numVertices;
      start_times[TOTAL] = omp_get_wtime();
      // deletions
      start_times[DELETION] = omp_get_wtime();
      {
        LogScore idelscore = i * del_score;
        for(int32_t v = 0; v < numVertices; v++) {
          LogScore match_value = match_sub<LogScore>(vertex_labels[v], read_base, match_score, subst_score);
          //edits[dst][v] = std::min(edits[src][v] + del_score, idelscore + match_value);
          edits_dst[v] = std::min(edits_src[v] + del_score, idelscore + match_value);
        }
      }
      times[DELETION] += omp_get_wtime() - start_times[DELETION];
      // matches and substitions
      start_times[MATCHMISMATCH] = omp_get_wtime();
      {
        for (int32_t u = 0; u < numVertices; u++) {
          for(int32_t e = offsets_out[u]; e < offsets_out[u+1]; e++) {
            int32_t v = edges[e];
            //edits[dst][v] = std::min(edits[dst][v], static_cast<LogScore>(edits[src][u] + match_sub<LogScore>(vertex_labels[v], read_base,match_score, subst_score)));
            edits_dst[v] = std::min(edits_dst[v], static_cast<LogScore>(edits_src[u] + match_sub<LogScore>(vertex_labels[v], read_base,match_score, subst_score)));
          }
        }
      }
      times[MATCHMISMATCH] += omp_get_wtime() - start_times[MATCHMISMATCH];
      // insertions assuming DAG
      start_times[INSERTION] = omp_get_wtime();
      {
        for (int32_t u = 0; u < numVertices; u++) {
          for(int32_t e = offsets_out[u]; e < offsets_out[u+1]; e++) {
            int32_t v = edges[e];
            if(! (match_cut && vertex_labels[v]==read[i]) ) {
              edits_dst[v] = std::min(edits_dst[v], static_cast<LogScore>(edits_dst[u] + ins_score));
            }
          }
        }
      }
      times[INSERTION] += omp_get_wtime() -  start_times[INSERTION];
      times[TOTAL] += omp_get_wtime() -  start_times[TOTAL];
    }

    LogScore min_score = edit_array[(readLength-1) * numVertices + 0]; // return scores at last character of read
    int32_t min_pos = 0;
    for(int32_t v = 0; v < numVertices; v++) {
      min_score = std::min(min_score, edit_array[(readLength-1) * numVertices + v]);
      if (min_score == edit_array[(readLength-1) * numVertices + v]) {
        min_pos = v;
      }
    }
    min_scores.push_back(min_score);
    std::cout << "read # " << num_reads_so_far+1 << ":";
    std::cout << "readLength is " << readLength << std::endl;
    std::cout << "min score is " << static_cast<int>(min_score) <<std::endl;
    //traceback
    std::string trace = "";
    trace.insert(0, 1, vertex_labels[min_pos]);
    size_t i = readLength-1;
    while(0 < i) {
      auto new_min_score = -1;
      int32_t new_min_pos = -1;
      size_t new_i = 0;
      // set edits_src and edits_dst
      edits_dst = edit_array + (i) * numVertices;
      edits_src = edit_array + (i-1) * numVertices;
      // insertions
      for (int32_t u = 0; u < numVertices; u++) {
        for(int32_t e = offsets_out[u]; e < offsets_out[u+1]; e++) {
          int32_t v = edges[e];
          if(v==min_pos) {
            if(! (match_cut && vertex_labels[v]==read[i]) ) {
              if(static_cast<LogScore>(edits_dst[u] + ins_score) == min_score) {
                new_min_score = edits_dst[u];
                new_min_pos = u;
                new_i = i;
              }
            }
          }
        }
      }
      // matches and substitions
      for (int32_t u = 0; u < numVertices; u++) {
        for(int32_t e = offsets_out[u]; e < offsets_out[u+1]; e++) {
          int32_t v = edges[e];
          if(v==min_pos) {
            if(static_cast<LogScore>(edits_src[u] + match_sub<LogScore>(vertex_labels[v], read[i],match_score, subst_score)) == min_score) {
              new_min_score = edits_src[u];
              new_min_pos = u;
              new_i = i-1;
            }
          }
        }
      }
      // deletions
      for(int32_t v = 0; v < numVertices; v++) {
        if(v==min_pos) {
          LogScore match_value = match_sub<LogScore>(vertex_labels[v], read[i], match_score, subst_score);
          if(std::min(edits_src[v] + del_score, (int)i * del_score + match_value) == min_score) {
            new_min_score = edits_src[v];
            new_min_pos = v;
          }
        }
      }
      min_score = new_min_score;
      i = new_i;
      if(min_pos != new_min_pos) {
        min_pos = new_min_pos;
        trace.insert(0, 1, vertex_labels[min_pos]);
      }
    }
    std::cout << "trace: " << trace << std::endl;
    num_reads_so_far++;
    for(int op_num = 0; op_num < NUM_OPERATIONS; op_num++) {
      serial_timing_calculations(times[op_num], num_reads_so_far, running_time[op_num], running_time_var[op_num]);
    }
  }
  for(int op_num = 0; op_num < NUM_OPERATIONS; op_num++) {
    // times
    if(num_reads_so_far<2) {
      stddev_time[op_num] = 0.0;
    } else {
      stddev_time[op_num] = std::sqrt(running_time_var[op_num]/(num_reads_so_far-1));
    }
  }
  delete [] edit_array;
  return min_scores;
}

template<typename LogScore>
inline void common_deletion_code(LogScore* edits[2], const bool &src, const bool &dst, const LogScore &del_score, const int &start, const int &end) {
  for(int32_t v = start; v < end; v++) {
    edits[dst][v] = edits[src][v] + del_score;
  }
}

template<typename LogScore>
std::vector<LogScore> serial_align_CSR(
    const std::vector<std::string> &reads,
    const int &numVertices,
    const std::vector<char> &vertex_labels,
    const std::vector<int> &offsets_out,
    const std::vector<int> &edges,
    const LogScore &match_score,
    const LogScore &subst_score,
    const LogScore &del_score,
    const LogScore &ins_score,
    std::vector<double> (&op_times)[NUM_OPERATIONS]) {

  LogScore* edit_array = new LogScore[2 * numVertices];
  LogScore* edits[2];
  std::vector<LogScore> min_scores;
  for (const std::string &read : reads) {
    edits[0] = edit_array;
    edits[1] = edit_array + numVertices;
    for(int32_t i = 0; i < numVertices; i++) {
      edits[0][i] = match_sub<LogScore>(vertex_labels[i], read[0], match_score, subst_score);
    }
    double times[NUM_OPERATIONS]  = {0.0};
    double start_times[NUM_OPERATIONS];
    auto readLength = read.length();
    for(size_t i = 1; i < readLength; i++) {
      char read_base = read[i];
      bool const src = (i - 1) & 1, dst = i & 1;
      start_times[TOTAL] = omp_get_wtime();
      // deletions
      start_times[DELETION] = omp_get_wtime();
      {
        common_deletion_code(edits,src,dst,del_score,0,numVertices);
      }
      times[DELETION] += omp_get_wtime() - start_times[DELETION];
      // matches and substitions
      start_times[MATCHMISMATCH] = omp_get_wtime();
      {
        for (int32_t u = 0; u < numVertices; u++) {
          for(int32_t e = offsets_out[u]; e < offsets_out[u+1]; e++) {
            int32_t v = edges[e];
            edits[dst][v] = std::min(edits[dst][v], static_cast<LogScore>(edits[src][u] + match_sub<LogScore>(vertex_labels[v], read_base,match_score, subst_score)));
          }
        }
      }
      times[MATCHMISMATCH] += omp_get_wtime() - start_times[MATCHMISMATCH];
      // insertions
      start_times[INSERTION] = omp_get_wtime();
      {
        if constexpr (dag) {
          // DAG: single topological pass
          for (int32_t u = 0; u < numVertices; ++u) {
            for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
              int32_t v = edges[e];
              edits[dst][v] = std::min(edits[dst][v],
                                      static_cast<LogScore>(edits[dst][u] + ins_score));
            }
          }
        } else {
          // Non-DAG
          std::vector<char> in_queue(numVertices, 0);
          std::vector<int> q;
          q.reserve(numVertices);

          for (int32_t u = 0; u < numVertices; ++u) {
            LogScore base_u = edits[dst][u];
            for (int32_t e = offsets_out[u]; e < offsets_out[u + 1]; ++e) {
              int32_t v = edges[e];
              LogScore possible = static_cast<LogScore>(base_u + ins_score);
              if (possible < edits[dst][v]) {
                edits[dst][v] = possible;
                if (!in_queue[v]) {
                  in_queue[v] = 1;
                  q.push_back(v);
                }
              }
            }
          }

          // Propagate improvements until no more changes
          for (size_t qi = 0; qi < q.size(); ++qi) {
            int x = q[qi];
            in_queue[x] = 0;
            LogScore base_x = edits[dst][x];
            for (int32_t e = offsets_out[x]; e < offsets_out[x + 1]; ++e) {
              int32_t v = edges[e];
              LogScore possible = static_cast<LogScore>(base_x + ins_score);
              if (possible < edits[dst][v]) {
                edits[dst][v] = possible;
                if (!in_queue[v]) {
                  in_queue[v] = 1;
                  q.push_back(v);
                }
              }
            }
          }
        }
      }
      times[INSERTION] += omp_get_wtime() - start_times[INSERTION];
      times[TOTAL] += omp_get_wtime() -  start_times[TOTAL];
    }

    LogScore min_score = edits[(readLength-1) & 1][0]; // return scores at last character of read
    for(int32_t v = 0; v < numVertices; v++) {
      min_score = std::min(min_score, edits[(readLength-1) & 1][v]);
    }
    for(int op_num = 0; op_num < NUM_OPERATIONS; op_num++) {
      op_times[op_num].push_back(times[op_num]);
    }
    min_scores.push_back(min_score);
  }
  delete [] edit_array;
  return min_scores;
}

// compare and swap without atomic datatype
template<typename LogScore>
// use pass by value for prev_value so original isn't modified
inline bool cas_update(LogScore* target, LogScore prev_value, LogScore &new_value) {
  while (new_value < prev_value) {
    LogScore old_value = prev_value;
    // Use __atomic_compare_exchange for CAS operation
    bool success = __atomic_compare_exchange(target, &old_value, &new_value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    if (success) {
        return true; // Update successful
    } else {
        prev_value = *target;
        new_value = std::min(new_value, prev_value);
    }
  }
  return false; // Update unsuccessful
}

template<typename LogScore>
inline void process_adjacency(const char &vertex_label, LogScore* edits[2], const char &read_base, const bool &src, const bool &dst, const int &match_score, const LogScore &subst_score, const int32_t &u, const int32_t &v) {
    LogScore prev_edits = edits[dst][v];
    LogScore new_edits = edits[src][u] + match_sub<LogScore>(vertex_label, read_base, match_score, subst_score);
    cas_update<LogScore>(&edits[dst][v], prev_edits, new_edits);
}

template<typename LogScore>
inline void vertex_parallel_match_mismatch_CSR(const int &numVertices, const std::vector<char> &vertex_label, const std::vector<int> &offsets_out, const std::vector<int> &adjcny_out, LogScore* edits[2], const char &read_base, const bool &src, const bool &dst, const int &match_score, const LogScore &subst_score) {
  #pragma omp for
  for (int32_t u = 0; u < numVertices; u++) {
    for(int32_t e = offsets_out[u]; e < offsets_out[u+1]; e++) {
      int32_t v = adjcny_out[e];
      process_adjacency<LogScore>(vertex_label[v], edits, read_base, src, dst, match_score, subst_score, u, v);
    }
  }
}

template<typename LogScore>
inline void insertion(const int &numVertices,uint8_t &thread_num, const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE], int32_t const read_base_index, const std::vector<int> (&component_indices)[ALPHABET_SIZE], const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE], const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE], LogScore* edits, const LogScore &ins_score) {
  auto &tco = thread_component_offsets[read_base_index];
  auto &comp_idx = component_indices[read_base_index];
  auto &offs = cut_offsets_out[read_base_index];
  auto &adj = cut_adjcny_out[read_base_index];

  // determine component range for this thread
  int component_after_this_thread = -1;
  for(int i = 1; component_after_this_thread == -1; i++) {
    component_after_this_thread = tco[thread_num+i];
  }


  if constexpr (dag) {
    for(int j = tco[thread_num]; j != -1 && j < component_after_this_thread; j++) {
      int u = comp_idx[j];
      for(int32_t e = offs[u]; e < offs[u+1]; e++) {
        int32_t v = adj[e];
        LogScore current_value = edits[v];
        LogScore possible_new_value = edits[u] + ins_score;
        if(possible_new_value < current_value) {
          edits[v] = possible_new_value;
        }
      }
    }  
  } else {
    // ---------- Thread-local queue + visited epochs ----------
    static thread_local std::vector<int> queue;
    static thread_local std::vector<uint32_t> in_queue_epoch;
    static thread_local uint32_t current_epoch = 1;

    // Resize once, never memset
    if (in_queue_epoch.size() < static_cast<size_t>(numVertices)) {
        in_queue_epoch.resize(numVertices, 0);
    }

    queue.clear();
    size_t qhead = 0;

    // ---------- Initial relaxation ----------
    for (int j = tco[thread_num]; j != -1 && j < component_after_this_thread; ++j) {
        int u = comp_idx[j];
        LogScore base = edits[u] + ins_score;

        for (int32_t e = offs[u]; e < offs[u + 1]; ++e) {
            int32_t v = adj[e];
            if (base < edits[v]) {
                edits[v] = base;
                if (in_queue_epoch[v] != current_epoch) {
                    in_queue_epoch[v] = current_epoch;
                    queue.push_back(v);
                }
            }
        }
    }

    // ---------- Propagation ----------
    while (qhead < queue.size()) {
        int x = queue[qhead++];
        LogScore base = edits[x] + ins_score;

        for (int32_t e = offs[x]; e < offs[x + 1]; ++e) {
            int32_t v = adj[e];
            if (base < edits[v]) {
                edits[v] = base;
                if (in_queue_epoch[v] != current_epoch) {
                    in_queue_epoch[v] = current_epoch;
                    queue.push_back(v);
                }
            }
        }
    }

    // Advance epoch (avoid overflow)
    if (++current_epoch == 0) {
        std::fill(in_queue_epoch.begin(), in_queue_epoch.end(), 0);
        current_epoch = 1;
    }
  }
}

double imbalance_calculation(double const &max_element, std::vector<double>& elements) {
    //double sum = std::reduce(elements.begin(), elements.end()); commented because it breaks valgrind
    if (elements.empty()) return 0; // Handle empty vector case
    double sum = 0.0;
    for (const auto& element : elements) {
        sum += element;
    }
    if (sum == 0) return max_element;
    return max_element / (sum / elements.size());
}

void parallel_timing_calculations(std::vector<double> &timing_vector, const int num_reads_so_far, double &running_time, double &running_time_var, double &running_imbalance, double &running_imbalance_var) {
  // welford method for one-pass variance
  double time = *std::max_element(timing_vector.begin(), timing_vector.end());
  double imbalance = imbalance_calculation(time, timing_vector);
  double old_time_mean = running_time;
  double old_imbalance_mean = running_imbalance;
  running_time += (time - old_time_mean) / num_reads_so_far;
  running_imbalance += (imbalance - old_imbalance_mean) / num_reads_so_far;
  running_time_var += (time-running_time)*(time-old_time_mean);
  running_imbalance_var += (imbalance-running_imbalance)*(imbalance-old_imbalance_mean);
}

template<typename LogScore>
std::vector<LogScore> parallel_align_CSR(
    const std::vector<std::string> &reads,
    const int numVertices,
    const std::vector<char> &vertex_labels,
    const std::vector<int> &offsets_out,
    const std::vector<int> &edges,
    const std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE],
    const std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE],
    const std::vector<int> (&component_indices)[ALPHABET_SIZE],
    const std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE],
    const LogScore match_score,
    const LogScore subst_score,
    const LogScore del_score,
    const LogScore ins_score,
    std::vector<double> (&op_times)[NUM_OPERATIONS]) {
  LogScore* edit_array = new LogScore[2 * numVertices];
  LogScore* edits[2];
  std::vector<int> vertex_parallel_thread_offsets = std::vector<int>(omp_get_max_threads()+1);
  // fill the thread-sized std::vectors serially
  for(uint8_t t = 0; t <= omp_get_max_threads(); t++) {
    vertex_parallel_thread_offsets[t] = get_thread_offset(t,omp_get_max_threads(), numVertices);
  }
  std::vector<LogScore> min_scores;
  for (const std::string &read : reads) {
    edits[0] = edit_array;
    edits[1] = edit_array + numVertices;
    #pragma omp parallel for
    for(int32_t i = 0; i < numVertices; i++) {
      edits[0][i] = match_sub<LogScore>(vertex_labels[i], read[0], match_score, subst_score);
    }
    std::vector<std::vector<double>> global_times(NUM_OPERATIONS,std::vector<double>(omp_get_max_threads(), 0.0)); // time on each stage of alignment on all threads, needs to be zeroed for each new read
    auto readLength = read.length();
    LogScore min_score = std::numeric_limits<LogScore>::max();
    #pragma omp parallel
    {
      uint8_t thread_num = omp_get_thread_num();
      double start_times[NUM_OPERATIONS];
      double private_times[NUM_OPERATIONS] = {0.0}; // time on each stage of alignment on owner thread
      for(size_t i = 1; i < readLength; i++) {
        start_times[TOTAL] = omp_get_wtime();
        char read_base = read[i];
        bool const src = (i - 1) & 1, dst = i & 1;
        // deletions
        start_times[DELETION] = omp_get_wtime();
        common_deletion_code(edits,src,dst,del_score,vertex_parallel_thread_offsets[omp_get_thread_num()],vertex_parallel_thread_offsets[omp_get_thread_num()+1]);
        private_times[DELETION] += (omp_get_wtime() - start_times[DELETION]);
        #pragma omp barrier
        // matches and substitions
        start_times[MATCHMISMATCH] = omp_get_wtime();
        vertex_parallel_match_mismatch_CSR<LogScore>(numVertices, vertex_labels, offsets_out, edges, edits, read_base, src, dst, match_score, subst_score);
        private_times[MATCHMISMATCH] += (omp_get_wtime() - start_times[MATCHMISMATCH]);
        #pragma omp barrier
        // insertions
        start_times[INSERTION] = omp_get_wtime();
        {
          bool const row = i & 1;
          int32_t const read_base_index = base_index(read_base);
          #if ALPHABET_SIZE == 4
          if(read_base_index != 4)
          #endif
          {
            insertion<LogScore>(numVertices, thread_num, thread_component_offsets, read_base_index, component_indices, cut_offsets_out, cut_adjcny_out, edits[row], ins_score);
          }
        }
        private_times[INSERTION] += omp_get_wtime() - start_times[INSERTION];
        private_times[TOTAL] += (omp_get_wtime() - start_times[TOTAL]);
        #pragma omp barrier
      }
      auto const result_row = (readLength-1) & 1;
      #pragma omp for reduction(min:min_score)
      for (int v = 0; v < numVertices; v++) {
          if (edits[result_row][v] < min_score) {
              min_score = edits[result_row][v];
          }
      }
      for(int op_num = 0; op_num < NUM_OPERATIONS; op_num++) {
        global_times[op_num][thread_num]=private_times[op_num];
      }
    }
    for(int op_num = 0; op_num < NUM_OPERATIONS; op_num++) {
      double max_time = 0.0;
      for (size_t t = 0; t < global_times[op_num].size(); ++t) {
        if (global_times[op_num][t] > max_time) max_time = global_times[op_num][t];
      }
      op_times[op_num].push_back(max_time);
    }
    min_scores.push_back(min_score);
  }
  delete [] edit_array;
  return min_scores;
}

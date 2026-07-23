#include <iostream>
#include <iomanip>
//#include "preprocessing.hpp"
// #include "csr_align.hpp"
#include <random> // for std::mt19937 and std::uniform_int_distribution
#include <fstream>
#include <sstream>
#include <string>
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include "pvector.h"
#include "graph.h"
#include "cc.cc"
#include "builder.h"
#include "config.hpp"

struct index_label_tuple {
    size_t index;
    int32_t label;
};
uint component_label(index_label_tuple i) {
    return i.label;
}
int32_t get_thread_offset(int32_t thread_num, int32_t num_threads, int32_t num_elements) {
    int32_t num_threads_w_extra_element = num_elements % num_threads;
    int32_t floor_elements_per_thread = num_elements / num_threads;
    // thread_offset is number of threads with extra vertex plus number of threads without extra vertex before current vertex
    int32_t current_thread_offset = (thread_num <= num_threads_w_extra_element) ? thread_num * (floor_elements_per_thread + 1) : num_threads_w_extra_element * (floor_elements_per_thread + 1) + (thread_num - num_threads_w_extra_element) * floor_elements_per_thread;
    return current_thread_offset;
}

struct Thread_offsets {
    int32_t current_thread_offset;
    int32_t next_thread_offset;
};
Thread_offsets get_thread_offsets(int32_t thread_num, int32_t num_threads, int32_t num_elements) {
    int32_t current_thread_offset = get_thread_offset(thread_num, num_threads, num_elements);
    int32_t next_thread_offset = (thread_num == num_threads - 1) ? num_elements
                                   : get_thread_offset(thread_num + 1, num_threads, num_elements);
    return {current_thread_offset, next_thread_offset};
}


template<typename... Args>
void print_variables(std::ofstream& out, Args&&... args) {
    std::ostringstream oss;
    ((oss << std::forward<Args>(args) << ","), ...);
    oss << "\n";
    out << oss.str();
}


double imbalance(double const &max_element, std::vector<double>& element_vector) {
    return max_element / (accumulate(element_vector.begin(), element_vector.end(), 0.0) / element_vector.size());
}

int32_t get_thread_component_offset(int32_t thread_num, int32_t num_threads, int32_t num_vertices, parlay::sequence<index_label_tuple> &tuples) {
    int32_t init = get_thread_offset(thread_num, num_threads, num_vertices);
    int i = 0;
    if(thread_num == 0) {
        return init;
    } else {
        int32_t current_thread_offset = init;
        int32_t prev_offset = get_thread_offset(thread_num-1, num_threads, num_vertices);
        int i = 0;
        while(i<current_thread_offset) {
            if(current_thread_offset - i <= prev_offset) {
                return -1;
            }
            if(init-i==0||(tuples)[init-i].label!=(tuples)[init-i-1].label) {
                return init-i;
            } else if ((tuples)[init+i].label!=(tuples)[init+i-1].label) {
                return init+i;
            }
            i++;
        }
        return -1;
    }
}

void label_and_sort_components(std::vector<Graph> &m, std::vector<parlay::sequence<index_label_tuple>> &tuples, std::vector<std::vector<int>> &thread_component_offsets, int& T) {
    // tuples = new parlay::sequence<index_label_tuple>[ALPHABET_SIZE];
    // thread_component_offsets = new int*[ALPHABET_SIZE];
    for(char base : bases) {
      pvector<int32_t> components = Afforest(m[base_index(base)], false);
      #ifdef DEBUG
      std::cout<<base;
      PrintCompStats(m[base_index(base)],components);
      #endif
      tuples[base_index(base)] = parlay::sequence<index_label_tuple>(components.size());
      parlay::parallel_for(0, components.size(), [&](size_t i) {
          tuples[base_index(base)][i] = {i, components[i]};
      });
      parlay::stable_integer_sort_inplace(tuples[base_index(base)], component_label);
      thread_component_offsets[base_index(base)] = std::vector<int>(T+1);
      thread_component_offsets[base_index(base)][T] = m[base_index(base)].num_nodes();

      parlay::parallel_for(0, T, [&](int t) {
          thread_component_offsets[base_index(base)][t] =
              get_thread_component_offset(
                  t, T,
                  m[base_index(base)].num_nodes(),
                  tuples[base_index(base)]);
      });
    }
}

CSRGraph<int32_t, int32_t, true> offsets_neighs_to_gapbs_csrgraph(pvector<int64_t>* offsets, int32_t *neighs, int32_t num_vertices) {
  int32_t **index = CSRGraph<int32_t, int32_t>::GenIndex(*offsets, neighs);
  CSRGraph<int32_t, int32_t, true> a(num_vertices, index, neighs);
  return a;
}

void match_cut_to_offsets_neighs(int32_t const numVertices, int32_t const numEdges, std::vector<int32_t> const adj_out, std::vector<int32_t> const offset_out, std::vector<char> const vertex_labels, bool const edge_parallel, std::vector<double> &match_cut_timing, pvector<int64_t> (&char_offset_out)[ALPHABET_SIZE], int32_t* (&char_adj_out)[ALPHABET_SIZE]) {
  int T = parlay::num_workers();
  // array to store each thread's local character counts
  pvector<int32_t> thread_char_counts[ALPHABET_SIZE];
  for(char base : bases) {
    thread_char_counts[base_index(base)] = pvector<int32_t>(T, 0);
  }
parlay::parallel_for(0, T, [&](int t) {
    auto thread_start_time = std::chrono::high_resolution_clock::now();
    if(edge_parallel) {
    } else {
      Thread_offsets t_o = get_thread_offsets(t, T, numVertices);
      int32_t local_char_counts[ALPHABET_SIZE] = {0};
      for (int32_t u = t_o.current_thread_offset; u < t_o.next_thread_offset; u++) {
        for(int32_t e = offset_out[u]; e < offset_out[u+1]; e++) {
          int8_t outward_neighbor_base_index = base_index(vertex_labels[adj_out[e]]);
          switch(outward_neighbor_base_index) {
            #if ALPHABET_SIZE == 4
            case 4: // if 'N' is scored as a match compared to unambiguous nucleotides
            for(uint8_t i = 0; i < 4; i++) {
              local_char_counts[i]++;
            }
            break;
            #endif
            case -1:
            throw std::runtime_error("invalid nucleotide: "+ vertex_labels[adj_out[e]]);
            break;
            default:
            local_char_counts[outward_neighbor_base_index]++;
          }
        }
      }
      for(char base : bases) {
        thread_char_counts[base_index(base)][t] = offset_out[t_o.next_thread_offset]-offset_out[t_o.current_thread_offset] - local_char_counts[base_index(base)];
      }
    }
    auto thread_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = thread_end_time - thread_start_time;
    match_cut_timing[t] += elapsed_time.count();
  });
  pvector<int64_t> thread_char_offsets[ALPHABET_SIZE];
  for(char base : bases) {
    char_offset_out[base_index(base)] = pvector<int64_t>(numVertices+1,0);
  }
  for(char base : bases) {
    thread_char_offsets[base_index(base)] = BuilderBase<int32_t, int32_t, int32_t, true>::ParallelPrefixSum(thread_char_counts[base_index(base)]);
    char_adj_out[base_index(base)] = new int32_t[thread_char_offsets[base_index(base)][T]]();
  }
  parlay::parallel_for(0, T, [&](int t) {
    auto thread_start_time = std::chrono::high_resolution_clock::now();
    if(edge_parallel) {
    } else {
      Thread_offsets t_o = get_thread_offsets(t, T, numVertices);
      //#pragma omp for
      for (int32_t u = t_o.current_thread_offset; u < t_o.next_thread_offset; u++) {
        for(char base : bases) {
          char_offset_out[base_index(base)][u] = thread_char_offsets[base_index(base)][t];
        }
        for(int32_t adj_index = offset_out[u]; adj_index < offset_out[u+1]; adj_index++) {
          int32_t v = adj_out[adj_index];
          for(char base : bases) {
            #if ALPHABET_SIZE == 4
            if('N' != vertex_labels[v] && base != vertex_labels[v]) {
            #else
            if(base != vertex_labels[v]) {
            #endif
              char_adj_out[base_index(base)][thread_char_offsets[base_index(base)][t]++] = v;
            }
          }
        }
      }
      if(t == T - 1) {
        for(char base : bases) char_offset_out[base_index(base)][numVertices] = thread_char_offsets[base_index(base)][T];
      }
    }
    auto thread_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = thread_end_time - thread_start_time;
    match_cut_timing[t] += elapsed_time.count();
  });
}

std::vector<Graph> match_cut_graphs(int32_t const numVertices, int32_t const numEdges, std::vector<int32_t> const adj_out, std::vector<int32_t> const offset_out, std::vector<char> const vertex_labels, bool const edge_parallel, std::vector<double> &match_cut_timing) {
  pvector<int64_t> char_offset_out[ALPHABET_SIZE];
  int32_t* char_adj_out[ALPHABET_SIZE];
  match_cut_to_offsets_neighs(numVertices, numEdges, adj_out, offset_out, vertex_labels, edge_parallel, match_cut_timing, char_offset_out, char_adj_out);

  std::vector<CSRGraph<int32_t, int32_t, true>> char_graphs;
  char_graphs.resize(ALPHABET_SIZE);
  // CSRGraph<int32_t, int32_t, true>* char_graphs = new CSRGraph<int32_t, int32_t, true>[ALPHABET_SIZE];
  for(char base : bases) {
    char_graphs[base_index(base)] = offsets_neighs_to_gapbs_csrgraph(&char_offset_out[base_index(base)],char_adj_out[base_index(base)],numVertices);
  }
  return char_graphs;
}

struct DiCharGraph {
    int numVertices = 0;
    int numEdges = 0;
    std::vector<char> vertex_label;      // single char label per vertex
    std::vector<int> offsets_out;        // size numVertices+1
    std::vector<int> adjcny_out;         // size numEdges
};

int main(int argc, char **argv) {
    if (argc < 3) {
        throw std::runtime_error("Provided less than 2 arguments");
    }
    DiCharGraph diCharGraph;
    bool match_cut;
    int match_score, subst_score, del_score, ins_score;
    double graph_loading_time;

    auto start_time = std::chrono::high_resolution_clock::now();
    {
        std::ifstream csr_fin(argv[1]);
        if (!csr_fin.is_open())
            throw std::runtime_error("Cannot open CSR file");

        int n = 0;
        int m = 0;
        csr_fin >> n >> m;
        diCharGraph.numVertices = n;
        diCharGraph.numEdges = m;

        // read labels: we expect one token per vertex; take the first character as the vertex label
        diCharGraph.vertex_label.resize(n);
        std::string label;
        for (int i = 0; i < n; ++i) {
            csr_fin >> label;
            if (label.empty()) diCharGraph.vertex_label[i] = 'N';
            else diCharGraph.vertex_label[i] = label[0];
        }

        // read (n+1) offsets: the snippet in the prompt wrote n offsets and then skipped the last;
        // here read n+1 offsets into offsets_out
        diCharGraph.offsets_out.resize(n + 1);
        for (int i = 0; i < n + 1; ++i) {
            int ofs;
            csr_fin >> ofs;
            diCharGraph.offsets_out[i] = ofs;
        }

        // read m destinations into adjcny_out
        diCharGraph.adjcny_out.resize(m);
        for (int i = 0; i < m; ++i) {
            int dst;
            csr_fin >> dst;
            diCharGraph.adjcny_out[i] = dst;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    graph_loading_time = elapsed_time.count();
    std::cout << "graph-loading time (s):\t" << graph_loading_time << '\n';
    // assert(diCharGraph.offsets_out.size() == diCharGraph.numVertices + 1);
    // assert(diCharGraph.offsets_out.back() == diCharGraph.adjcny_out.size());
    std::string output_filename(argv[2]);

    bool const edge_parallel = false;
    int T = parlay::num_workers();
    std::vector<double> match_cut_timing(T, 0);
    double components_timing = 0;
    std::vector<parlay::sequence<index_label_tuple>> tuples(ALPHABET_SIZE);
    // parlay::sequence<index_label_tuple> *tuples;
    double match_cut_time = 0;
    std::vector<std::vector<int>> thread_component_offsets(ALPHABET_SIZE);
    // int **thread_component_offsets;
    double preprocessing_time = 0;
    pvector<int64_t> char_offset_out[ALPHABET_SIZE];
    int32_t* char_adj_out[ALPHABET_SIZE];
    std::vector<Graph> m;

    std::vector<double> match_cut_to_offsets_timing(T, 0);

    double match_cut_to_offsets_time = 0;
    double match_cut_graphs_time = 0;
    double label_and_sort_time = 0;

    auto preprocessing_start_time = std::chrono::high_resolution_clock::now();

    // --- match_cut_to_offsets_neighs timing ---
    {
      auto start_time = std::chrono::high_resolution_clock::now();
      match_cut_to_offsets_neighs(
          diCharGraph.numVertices,
          diCharGraph.numEdges,
          diCharGraph.adjcny_out,
          diCharGraph.offsets_out,
          diCharGraph.vertex_label,
          edge_parallel,
          match_cut_to_offsets_timing, // function may write per-thread times
          char_offset_out,
          char_adj_out);
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;
      match_cut_to_offsets_time = elapsed.count();
    }

    // --- match_cut_graphs timing ---
    match_cut_timing.assign(T, 0); // ensure zeroed for the call
    {
      auto start_time = std::chrono::high_resolution_clock::now();
      m = match_cut_graphs(
          diCharGraph.numVertices,
          diCharGraph.numEdges,
          diCharGraph.adjcny_out,
          diCharGraph.offsets_out,
          diCharGraph.vertex_label,
          edge_parallel,
          match_cut_timing); // function may write per-thread times
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;
      match_cut_graphs_time = elapsed.count();
    }

    // --- label_and_sort_components timing ---
    {
      auto start_time = std::chrono::high_resolution_clock::now();
      label_and_sort_components(m, tuples, thread_component_offsets, T);
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;
      label_and_sort_time = elapsed.count();
    }

    // --- overall preprocessing timing ---
    auto preprocessing_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> preprocessing_elapsed_time = preprocessing_end_time - preprocessing_start_time;
    preprocessing_time = preprocessing_elapsed_time.count();

    // --- Print results (no per-thread prints) ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "character graph construction time (s):\t" << preprocessing_time << '\n';
    std::cout << "match_cut_to_offsets_neighs time (s):\t" << match_cut_to_offsets_time << '\n';
    std::cout << "match_cut_graphs time (s):\t\t" << match_cut_graphs_time << '\n';
    std::cout << "label_and_sort_components time (s):\t" << label_and_sort_time << '\n';

    // std::ofstream cut_output;
    // std::ofstream component_output;
    // double char_graph_writing_time;
    // start_time = omp_get_wtime();
    // for(char base : bases) {
    //     cut_output.open(output_filename+"-"+base+".csr");
    //     component_output.open(output_filename+"-"+base+".components");
    //     cut_output << diCharGraph.numVertices << '\n';
    //     cut_output << char_offset_out[base_index(base)][diCharGraph.numVertices]  << '\n';
    //     for(int i = 0; i < diCharGraph.numVertices+1; i++) {
    //         cut_output << char_offset_out[base_index(base)][i] << " "; 
    //     }
    //     cut_output << "\n";
    //     for(int i = 0; i < char_offset_out[base_index(base)][diCharGraph.numVertices]; i++) {
    //         cut_output << char_adj_out[base_index(base)][i] << " "; 
    //     }
    //     cut_output << "\n";
    //     cut_output.close();
    //     for(int i = 0; i < tuples[base_index(base)].size(); i++) {
    //         component_output << tuples[base_index(base)][i].index << " ";
    //     }
    //     component_output << '\n';
    //     for(int i = 0; i < tuples[base_index(base)].size(); i++) {
    //         component_output << tuples[base_index(base)][i].label << " ";
    //     }
    //     component_output << '\n';
    //     component_output.close();
    //     int const max_threads = parlay::num_workers();
    //     for (int num_threads = 1; num_threads <= max_threads; num_threads = ((num_threads <= max_threads/ 2) || (num_threads == max_threads)) ? (num_threads *= 2) : max_threads) {
    //         std::ofstream thread_component_offsets_output;
    //         thread_component_offsets_output.open(output_filename+"-"+base+"-"+std::to_string(num_threads)+".components");
    //         omp_set_num_threads(num_threads);
    //         label_and_sort_components(m,tuples,thread_component_offsets, num_threads);
    //         for(int i = 0; i < num_threads+1; i++) {
    //             thread_component_offsets_output << thread_component_offsets[base_index(base)][i] << " ";
    //         }
    //         thread_component_offsets_output << '\n';
    //         thread_component_offsets_output.close();
    //     }
    // }
    // char_graph_writing_time = omp_get_wtime() - start_time;
    // std::cout << "character graph writing time (s):\t" << char_graph_writing_time << '\n';
    // double max_cut_time = *std::max_element(match_cut_timing.begin(), match_cut_timing.end());
    // double cut_imbalance = imbalance(max_cut_time, match_cut_timing);

    // std::cout << "not timed: counting number of components per character graph" << std::endl;
    // for(char base : bases) {
    //   std::cout << base << std::endl;
    //   pvector<int32_t> components = Afforest(m[base_index(base)], false);
    //   PrintCompStats(m[base_index(base)],components);
    // }
    return 0;
}
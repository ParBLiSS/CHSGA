#include <map>
#include <iostream>
#include "graphLoad.hpp"
#include "cc.cc"
#include "builder.h"
#include "graph.h"
#include "pvector.h"
#include <parlay/parallel.h>
#include <parlay/sequence.h>
#include <parlay/primitives.h>


int32_t get_thread_component_offset(int32_t thread_num, int32_t num_threads, int32_t num_vertices, parlay::sequence<index_label_tuple> *tuples) {
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
            if(init-i==0||(*tuples)[init-i].label!=(*tuples)[init-i-1].label) {
                return init-i;
            } else if ((*tuples)[init+i].label!=(*tuples)[init+i-1].label) {
                return init+i;
            }
            i++;
        }
        return -1;
    }
}

void label_and_sort_components(Graph* m, parlay::sequence<index_label_tuple>* &tuples, int **&thread_component_offsets) {
    tuples = new parlay::sequence<index_label_tuple>[ALPHABET_SIZE];
    thread_component_offsets = new int*[ALPHABET_SIZE];
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
      thread_component_offsets[base_index(base)] = new int[omp_get_max_threads()+1];
      thread_component_offsets[base_index(base)][omp_get_max_threads()] = m->num_nodes();
      #pragma omp parallel
      {
          thread_component_offsets[base_index(base)][omp_get_thread_num()] = get_thread_component_offset(omp_get_thread_num(), omp_get_num_threads(), m->num_nodes(), &tuples[base_index(base)]);
      }
    }
}

CSRGraph<int32_t, int32_t, true> offsets_neighs_to_gapbs_csrgraph(pvector<int64_t>* offsets, int32_t *neighs, int32_t num_vertices) {
  int32_t **index = CSRGraph<int32_t, int32_t>::GenIndex(*offsets, neighs);
  CSRGraph<int32_t, int32_t, true> a(num_vertices, index, neighs);
  return a;
}

void match_cut_to_offsets_neighs(int32_t const numVertices, int32_t const numEdges, std::vector<int32_t> const adj_out, std::vector<int32_t> const offset_out, std::vector<char> const vertex_labels, bool const edge_parallel, std::vector<double> &match_cut_timing, pvector<int64_t> (&char_offset_out)[ALPHABET_SIZE], int32_t* (&char_adj_out)[ALPHABET_SIZE]) {
  // array to store each thread's local character counts
  pvector<int32_t> thread_char_counts[ALPHABET_SIZE];
  for(char base : bases) {
    thread_char_counts[base_index(base)] = pvector<int32_t>(omp_get_max_threads(),0);
  }
  #pragma omp parallel
  {
    double thread_start_time = omp_get_wtime();
    if(edge_parallel) {
    } else {
      Thread_offsets t_o = get_thread_offsets(omp_get_thread_num(),omp_get_num_threads(), numVertices);
      int32_t local_char_counts[ALPHABET_SIZE] = {0};
      for (int32_t u = t_o.current_thread_offset; u < t_o.next_thread_offset; u++) {
        for(int32_t e = offset_out[u]; e < offset_out[u+1]; e++) {
          int8_t outward_neighbor_base_index = base_index(vertex_labels[adj_out[e]]);
          switch(outward_neighbor_base_index) {
            case 4:
            for(uint8_t i = 0; i < 4; i++) {
              local_char_counts[i]++;
            }
            break;
            case -1:
            throw std::runtime_error("invalid nucleotide: "+ vertex_labels[adj_out[e]]);
            break;
            default:
            local_char_counts[outward_neighbor_base_index]++;
          }
        }
      }
      for(char base : bases) {
        thread_char_counts[base_index(base)][omp_get_thread_num()] = offset_out[t_o.next_thread_offset]-offset_out[t_o.current_thread_offset] - local_char_counts[base_index(base)];
      }
    }
    match_cut_timing[omp_get_thread_num()] += (omp_get_wtime() - thread_start_time);
  }
  pvector<int64_t> thread_char_offsets[ALPHABET_SIZE];
  for(char base : bases) {
    char_offset_out[base_index(base)] = pvector<int64_t>(numVertices+1,0);
  }
  for(char base : bases) {
    thread_char_offsets[base_index(base)] = BuilderBase<int32_t, int32_t, int32_t, true>::ParallelPrefixSum(thread_char_counts[base_index(base)]);
    char_adj_out[base_index(base)] = new int32_t[thread_char_offsets[base_index(base)][omp_get_max_threads()]](); // initializes to 0s
  }
  #pragma omp parallel
  {
    double thread_start_time = omp_get_wtime();
    if(edge_parallel) {
    } else {
      Thread_offsets t_o = get_thread_offsets(omp_get_thread_num(),omp_get_num_threads(), numVertices);
      //#pragma omp for
      for (int32_t u = t_o.current_thread_offset; u < t_o.next_thread_offset; u++) {
        for(char base : bases) {
          char_offset_out[base_index(base)][u] = thread_char_offsets[base_index(base)][omp_get_thread_num()];
        }
        for(int32_t adj_index = offset_out[u]; adj_index < offset_out[u+1]; adj_index++) {
          int32_t v = adj_out[adj_index];
          for(char base : bases) {
            if('N' != vertex_labels[v] && base != vertex_labels[v]) {
              char_adj_out[base_index(base)][thread_char_offsets[base_index(base)][omp_get_thread_num()]++] = v;
            }
          }
        }
      }
      if(omp_get_thread_num()==omp_get_num_threads()-1) {
        for(char base : bases) char_offset_out[base_index(base)][numVertices] = thread_char_offsets[base_index(base)][omp_get_max_threads()];
      }
    }
    match_cut_timing[omp_get_thread_num()] += (omp_get_wtime() - thread_start_time);
  }
}

Graph *match_cut_graphs(int32_t const numVertices, int32_t const numEdges, std::vector<int32_t> const adj_out, std::vector<int32_t> const offset_out, std::vector<char> const vertex_labels, bool const edge_parallel, std::vector<double> &match_cut_timing) {
  pvector<int64_t> char_offset_out[ALPHABET_SIZE];
  int32_t* char_adj_out[ALPHABET_SIZE];
  match_cut_to_offsets_neighs(numVertices, numEdges, adj_out, offset_out, vertex_labels, edge_parallel, match_cut_timing, char_offset_out, char_adj_out);

  CSRGraph<int32_t, int32_t, true>* char_graphs = new CSRGraph<int32_t, int32_t, true>[ALPHABET_SIZE];
  for(char base : bases) {
    char_graphs[base_index(base)] = offsets_neighs_to_gapbs_csrgraph(&char_offset_out[base_index(base)],char_adj_out[base_index(base)],numVertices);
  }
  return char_graphs;
}

CSRGraph<int32_t, int32_t, true> pasgal_to_gapbs_csrgraph(psgl::CSR_char_container diCharGraph, pvector<int64_t>** gapbs_offsets) {
    *gapbs_offsets = new pvector<int64_t>(diCharGraph.offsets_out.size(),0);
    #pragma omp parallel for
    for (size_t i=0; i < diCharGraph.offsets_out.size(); i++)
      (**gapbs_offsets)[i] = diCharGraph.offsets_out[i];
    int32_t *neighs = new int32_t[diCharGraph.numEdges];
    #pragma omp parallel for
    for(int32_t e = 0; e < diCharGraph.numEdges; e++) {
      neighs[e] = diCharGraph.adjcny_out[e];
    }
    return offsets_neighs_to_gapbs_csrgraph(*gapbs_offsets, neighs, diCharGraph.numVertices);
}
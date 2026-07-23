#include <map>
#include <iostream>

#include "config.hpp"

struct index_label_tuple {
    uint32_t index;
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
Thread_offsets get_thread_offsets(int32_t thread_num, int32_t num_threads, int32_t num_edges) {
  int32_t current_thread_offset = get_thread_offset(thread_num, num_threads, num_edges);
  int32_t next_thread_offset = get_thread_offset(thread_num + 1, num_threads, num_edges);
  return {current_thread_offset, next_thread_offset};
}

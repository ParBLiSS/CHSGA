#include "build_pch.hpp"

#include "dijkstra.hpp"
#include "query.hpp"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr,
            "WeightedGraph only\n"
            "Usage: %s [-i input_graph]\n"
            "Options:\n"
            "\t-o,\toutput_ch_graph\n"
            "\t-p,\tmax_pop_count\n"
            "\t-s,\tselect fraction\n"
            "\t-t,\ts-t query verify numn"
            "\t-q,\tsssp query verify num\n"
            "\t-b,\tdegree bound\n"
            "\t-d,\tprint per round detail\n",
            argv[0]);
    return 0;
  }
  char c;
  int max_pop_count = 500;
  int bidirect_verify_num = 0;
  int sssp_verify_num = 0;
  int degree_bound = 0;
  bool degree_bounded = false, print_detail = false, write_ch = false;
  double sample_bound = 1;
  EdgeTy k_value = std::numeric_limits<EdgeTy>::max();
  std::vector<size_t> vertex_label_offsets = {};
  while ((c = getopt(argc, argv, "i:o:p:s:t:q:b:dk:v:")) != -1) {
    switch (c) {
      case 'i':
        INPUT_FILEPATH = optarg;
        break;
      case 'o':
        write_ch= true;
        OUTPUT_FILEPATH = optarg;
        break;
      case 'p':
        max_pop_count = atol(optarg);
        if (max_pop_count < 0) {
          fprintf(stderr, "Error: max_pop_count must be non negative\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 's':
        sample_bound = atof(optarg);
        if (sample_bound <= 0 || sample_bound >1) {
          fprintf(stderr, "Error: selection_fraction must be larger than 0 and smaller than or equal to 1\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 't':
        bidirect_verify_num = atol(optarg);
        if (bidirect_verify_num < 0) {
          fprintf(stderr, "Error: bidirect_verify_num must be non negative\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 'q':
        sssp_verify_num = atol(optarg);
        if (sssp_verify_num < 0) {
          fprintf(stderr, "Error: sssp_verify_num must be non negative\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 'b':
        degree_bounded = true;
        degree_bound = atol(optarg);
        if (degree_bound < 0) {
          fprintf(stderr, "Error: degree_bound must be non negative\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 'd':
        print_detail = true;
        break;
      case 'k':
        k_value = atof(optarg);  // Parse k as a floating point value (EdgeTy is assumed to be a float/double)
        if (k_value < 0) {
          fprintf(stderr, "Error: k (shortcut weight bound) must be non-negative\n");
          exit(EXIT_FAILURE);
        }
      case 'v':
        LABEL_OFFSETS_FILEPATH = optarg;
        break;
      break;
      default:
        fprintf(stderr, "Error: Unknown option %c\n", optopt);
        exit(EXIT_FAILURE);
    }
  }
  Graph origin_graph = read_graph(INPUT_FILEPATH);

  if (LABEL_OFFSETS_FILEPATH != nullptr) {
    std::ifstream label_offsets_file(LABEL_OFFSETS_FILEPATH);  // Open the file for reading
    if (!label_offsets_file) {
        std::cerr << "Error opening vertex labels file." << std::endl;
    } else {
        size_t label_offset;
        while (label_offsets_file >> label_offset) { // Read space-separated sequences as single strings
            vertex_label_offsets.push_back(label_offset);
        }
        label_offsets_file.close();
    }
  }
  std::vector<size_t> label_lengths;
  if (!vertex_label_offsets.empty()) {
      label_lengths.reserve(vertex_label_offsets.size() - 1); // Reserve for efficiency
      for (size_t i = 1; i < vertex_label_offsets.size(); ++i) {
          label_lengths.push_back(vertex_label_offsets[i] - vertex_label_offsets[i - 1]);
      }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  PCH *solver =
      new PCH(origin_graph, max_pop_count, degree_bounded, degree_bound, sample_bound, k_value, print_detail, label_lengths);
  PchGraph contracted_graph = solver->createContractionHierarchy();
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[ build time ] : " << sec << " sec\n";
  delete (solver);
  if(write_ch) {
    auto t0 = std::chrono::high_resolution_clock::now();
    write_pbbs_format(contracted_graph, OUTPUT_FILEPATH);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[ write time ] : " << sec << " sec\n";
  }
  return 0;
}
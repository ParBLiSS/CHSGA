#include "align.hpp"
#include <getopt.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <variant>
#include <string>
#include <stdexcept>
#include <iomanip>
#include "load_reads.hpp"



// Buckets for |u-v| thresholds; tune as needed
static const std::array<size_t, 20> SPAN_THRESH = {
  1, 2, 4, 8, 16, 32, 64, 128, 256,
  512, 1024, 2048, 4096, 8192, 16384,
  32768, 65536, 131072, 262144, 524288
};

void measure_edge_spans(const Graph& G) {
  std::vector<uint64_t> counts(SPAN_THRESH.size()+1, 0);
  uint64_t total_edges = 0;
  uint64_t max_span = 0;

  for (NodeId u = 0; u < G.n; ++u) {
    for (EdgeId e = G.offset[u]; e < G.offset[u+1]; ++e) {
      NodeId v = G.E[e].v;
      size_t span = (u > v) ? (u - v) : (v - u);
      max_span = std::max<uint64_t>(max_span, span);
      ++total_edges;

      size_t b = 0;
      while (b < SPAN_THRESH.size() && span > SPAN_THRESH[b]) ++b;
      ++counts[b];
    }
  }

  std::cout << "=== Edge span histogram ===\n";
  std::cout << "Total edges: " << total_edges << " Max span: " << max_span << "\n";
  for (size_t i = 0; i < SPAN_THRESH.size(); ++i) {
    double pct = 100.0 * (double)counts[i] / (double)total_edges;
    std::cout << "  |u-v| <= " << std::setw(6) << SPAN_THRESH[i] << " : "
              << std::fixed << std::setprecision(2) << pct << "%\n";
  }
  double pct_tail = 100.0 * (double)counts.back() / (double)total_edges;
  std::cout << "  |u-v|  > " << std::setw(6) << SPAN_THRESH.back() << " : "
            << std::fixed << std::setprecision(2) << pct_tail << "%\n";
}
void measure_degree_histograms(const Graph& G) {
  std::vector<size_t> indeg(G.n, 0);
  std::vector<size_t> outdeg(G.n, 0);

  for (NodeId u = 0; u < G.n; ++u) {
    size_t od = G.offset[u+1] - G.offset[u];
    outdeg[u] = od;
    for (EdgeId e = G.offset[u]; e < G.offset[u+1]; ++e) {
      ++indeg[G.E[e].v];
    }
  }

  auto bucketize = [](const std::vector<size_t>& deg) {
    // buckets: 0,1,2,3-4,5-8,9-16,17-32,>32
    std::array<uint64_t,8> b{}; b.fill(0);
    for (auto d : deg) {
      if (d == 0) ++b[0];
      else if (d == 1) ++b[1];
      else if (d == 2) ++b[2];
      else if (d <= 4) ++b[3];
      else if (d <= 8) ++b[4];
      else if (d <= 16) ++b[5];
      else if (d <= 32) ++b[6];
      else ++b[7];
    }
    return b;
  };

  auto bi = bucketize(indeg);
  auto bo = bucketize(outdeg);

  auto printBuckets = [](const char* name, const std::array<uint64_t,8>& b, size_t n) {
    std::cout << "=== " << name << " histogram ===\n";
    const char* labels[8] = {"0","1","2","3-4","5-8","9-16","17-32",">32"};
    for (int i = 0; i < 8; ++i) {
      double pct = 100.0 * (double)b[i] / (double)n;
      std::cout << "  " << std::setw(5) << labels[i] << ": "
                << std::fixed << std::setprecision(2) << pct << "%\n";
    }
  };
  printBuckets("In-degree", bi, G.n);
  printBuckets("Out-degree", bo, G.n);

  size_t max_in = *std::max_element(indeg.begin(), indeg.end());
  size_t max_out = *std::max_element(outdeg.begin(), outdeg.end());
  std::cout << "Max in-degree: " << max_in << " Max out-degree: " << max_out << "\n";
}
// For a chosen tile size T, measure fraction of edges where both u and v fall in the same tile by dest.
// Tile index by destination node ID; adjust if you use label positions instead.
void measure_tile_capture(const Graph& G, size_t T) {
  uint64_t captured = 0, total = 0;
  for (NodeId u = 0; u < G.n; ++u) {
    for (EdgeId e = G.offset[u]; e < G.offset[u+1]; ++e) {
      NodeId v = G.E[e].v;
      size_t tile_u = u / T;
      size_t tile_v = v / T;
      if (tile_u == tile_v) ++captured;
      ++total;
    }
  }
  double pct = 100.0 * (double)captured / (double)total;
  std::cout << "Tile size " << T << ": captured " << std::fixed << std::setprecision(2)
            << pct << "% of edges in same tile\n";
}


int main(int argc, char** argv) {
  char* orig_char_fname   = nullptr;
  char* orig_str_fname    = nullptr;
  char* contr_str_fname   = nullptr;
  char* reads_fname       = nullptr;
  char* labels_fname      = nullptr;
  char* offsets_fname     = nullptr;


  int opt;
  while ((opt = getopt(argc, argv, "I:C:i:c:r:l:o:a:s:N:M:y:")) != -1) {
    switch (opt) {

      // new origin/contracted args
      case 'I': orig_char_fname  = optarg; break;
      case 'i': orig_str_fname = optarg; break;
      case 'c': contr_str_fname  = optarg; break;
      case 'r': reads_fname  = optarg; break;
      case 'l': labels_fname = optarg; break;
      case 'o': offsets_fname= optarg; break;

      default:
        std::cerr << "Usage: " << argv[0]
                  << " -I orig_char -i orig_string -c contr_string"
                  " -r reads_file -l labels_file -o offsets_file"
                  " \n";
        return EXIT_FAILURE;
    }
  }

  // otherwise enforce original mode prerequisites (now require both char and string graphs)
  if (!orig_char_fname || !orig_str_fname ||
      !contr_str_fname ||
      !reads_fname  || !labels_fname || !offsets_fname) {
    std::cerr << "Error: missing required arguments\n";
    return EXIT_FAILURE;
  }
  std::vector<std::string> reads = load_reads(reads_fname);
  double input_start_time = omp_get_wtime();
  //
  // Read vertex‐labels (shared labels file)
  //
  std::vector<char> vertex_labels;
  {
    std::ifstream ifs(labels_fname);
    if (!ifs.is_open()) {
      std::cerr << "Cannot open labels file: " << labels_fname << "\n";
      return EXIT_FAILURE;
    }
    char ch;
    while (ifs.get(ch))
      if (ch != '\n')
        vertex_labels.push_back(ch);
  }

  //
  // Read label‐offsets (for STRING mapper)
  //
  std::vector<size_t> vertex_label_offsets;
  {
    std::ifstream ifs(offsets_fname);
    if (!ifs.is_open()) {
      std::cerr << "Cannot open offsets file: " << offsets_fname << "\n";
      return EXIT_FAILURE;
    }
    size_t off;
    while (ifs >> off)
      vertex_label_offsets.push_back(off);
  }

  //
  // Load graphs: origin and contracted for both CHAR and STRING
  //
  Graph origin_char_graph = read_graph(orig_char_fname);
  InCSR inGraph = build_in_csr(std::move(origin_char_graph));
  Graph origin_string_graph = read_graph(orig_str_fname);

  PchGraph contracted_string_graph;
  double input_ch_start_time = omp_get_wtime();
  read_contracted_graph(contr_str_fname, contracted_string_graph);
  PchQuery query(contracted_string_graph, origin_string_graph);
  double input_ch_time = omp_get_wtime() - input_ch_start_time;
  std::cout << "input ch time: " << input_ch_time << '\n';
  double input_time = omp_get_wtime() - input_start_time;
  std::cout << "input time: " << input_time << '\n' << std::flush;

  sequence<NodeId> contracted_str_to_og(contracted_string_graph.n);
  parallel_for(0, contracted_string_graph.n, [&](size_t i) {
    contracted_str_to_og[ contracted_string_graph.rank[i] ] = i;
  });

  //
  // Prepare two mappers: one for CHAR (no offsets) and one for STRING (uses offsets)
  //
  VertexLabelMapper<true>  mapper_str(vertex_label_offsets);

  //
  // Scoring parameters (unchanged)
  //
  int match_score = 0, subst_score = 1, del_score = 1, ins_score = 1;
  std::vector<int> scores;
  auto t0 = std::chrono::high_resolution_clock::now();
  scores = align_combo(
    reads,
    vertex_labels,
    inGraph,
    origin_string_graph,
    contracted_str_to_og,
    match_score, subst_score, del_score, ins_score, query,
    mapper_str
  );
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
  std::cout << "[total over " << reads.size() << " reads] : " << ms << " ms\n";
  printVector("score", scores);
  return 0;
}

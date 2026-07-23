// ConSGA_traceback.cpp
// Drop-in replacement for ConSGA.cpp that runs checkpointed traceback.
//
// Usage (same flags as ConSGA):
//   ConSGA_traceback_mismatchN_dag \
//     -I orig_char.adj   -i orig_string.adj   -c contracted.adj \
//     -r reads.fastq     -l labels.txt        -o offsets.txt
//
// Optional flags:
//   -b          benchmark mode: time score+traceback only (paper-style output)
//   -k <int>    checkpoint every k rows    (default 500)
//   -d <dir>    NVMe directory for ckpts   (default /tmp/consga_traceback_ckpts)

#include "traceback.hpp"
#include <getopt.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <cmath>
#include "load_reads.hpp"

static void print_mean_with_moe(const char* label, const std::vector<double>& times) {
    if (times.size() <= 1) {
        std::cout << " " << label << ":        "
                  << (times.empty() ? 0.0 : times[0]) << " sec\n";
        return;
    }
    std::vector<double> trimmed(times.begin() + 1, times.end());
    double mean = 0.0;
    for (double t : trimmed) mean += t;
    mean /= static_cast<double>(trimmed.size());

    double var = 0.0;
    for (double t : trimmed) {
        double d = t - mean;
        var += d * d;
    }
    var /= static_cast<double>(trimmed.size() - 1);
    double se = std::sqrt(var / static_cast<double>(trimmed.size()));
    double moe = 1.96 * se;
    int df = static_cast<int>(trimmed.size()) - 1;

    std::cout << " " << label << ":        "
              << std::fixed << std::setprecision(4) << mean
              << " ± " << moe << " sec (n=" << trimmed.size()
              << ", df=" << df << ")\n";
}

int main(int argc, char** argv) {
    char* orig_char_fname  = nullptr;
    char* orig_str_fname   = nullptr;
    char* contr_str_fname  = nullptr;
    char* reads_fname      = nullptr;
    char* labels_fname     = nullptr;
    char* offsets_fname    = nullptr;

    CheckpointConfig cfg;
    cfg.k       = 500;
    cfg.dir     = "/tmp/consga_traceback_ckpts";
    cfg.cleanup = true;
    bool benchmark = false;

    int opt;
    while ((opt = getopt(argc, argv, "bI:i:c:r:l:o:k:d:")) != -1) {
        switch (opt) {
            case 'b': benchmark = true; break;
            case 'I': orig_char_fname = optarg; break;
            case 'i': orig_str_fname  = optarg; break;
            case 'c': contr_str_fname = optarg; break;
            case 'r': reads_fname     = optarg; break;
            case 'l': labels_fname    = optarg; break;
            case 'o': offsets_fname   = optarg; break;
            case 'k': cfg.k           = std::stoll(optarg); break;
            case 'd': cfg.dir         = optarg; break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [-b] -I orig_char -i orig_string -c contracted"
                             " -r reads -l labels -o offsets"
                             " [-k checkpoint_every_k_rows] [-d ckpt_dir]\n";
                return EXIT_FAILURE;
        }
    }

    if (!orig_char_fname || !orig_str_fname || !contr_str_fname ||
        !reads_fname || !labels_fname || !offsets_fname) {
        std::cerr << "Error: missing required arguments\n";
        return EXIT_FAILURE;
    }

    std::vector<std::string> reads = load_reads(reads_fname);

    std::vector<char> vertex_labels;
    {
        std::ifstream ifs(labels_fname);
        if (!ifs) { std::cerr << "Cannot open labels: " << labels_fname << "\n"; return EXIT_FAILURE; }
        char ch;
        while (ifs.get(ch))
            if (ch != '\n') vertex_labels.push_back(ch);
    }

    std::vector<size_t> vertex_label_offsets;
    {
        std::ifstream ifs(offsets_fname);
        if (!ifs) { std::cerr << "Cannot open offsets: " << offsets_fname << "\n"; return EXIT_FAILURE; }
        size_t off;
        while (ifs >> off) vertex_label_offsets.push_back(off);
    }

    Graph origin_char_graph   = read_graph(orig_char_fname);
    InCSR inGraph             = build_in_csr(std::move(origin_char_graph));
    Graph origin_string_graph = read_graph(orig_str_fname);

    PchGraph contracted_string_graph;
    read_contracted_graph(contr_str_fname, contracted_string_graph);
    PchQuery query(contracted_string_graph, origin_string_graph);

    sequence<NodeId> contracted_str_to_og(contracted_string_graph.n);
    parallel_for(0, contracted_string_graph.n, [&](size_t i) {
        contracted_str_to_og[ contracted_string_graph.rank[i] ] = i;
    });

    VertexLabelMapper<true> mapper_str(vertex_label_offsets);

    const int match_score = 0, subst_score = 1, del_score = 1, ins_score = 1;

    if (benchmark) {
        std::vector<double> per_read_sec;
        per_read_sec.reserve(reads.size());

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t ri = 0; ri < reads.size(); ++ri) {
            CheckpointConfig rcfg = cfg;
            rcfg.dir = cfg.dir + "/read_" + std::to_string(ri);

            auto r0 = std::chrono::high_resolution_clock::now();
            (void) align_read_with_traceback<int>(
                reads[ri], vertex_labels.size(),
                vertex_labels, inGraph,
                origin_string_graph, contracted_str_to_og,
                match_score, subst_score, del_score, ins_score,
                query, mapper_str, rcfg);
            auto r1 = std::chrono::high_resolution_clock::now();
            per_read_sec.push_back(
                std::chrono::duration<double>(r1 - r0).count());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "Average times (skipping first read), 95% CI:\n";
        print_mean_with_moe(" Total", per_read_sec);
        std::cout << " Substitutions: NA sec\n";
        std::cout << " Insertions:    NA sec\n";
        std::cout << "[total over " << reads.size() << " reads] : " << ms << " ms\n";
        return 0;
    }

    // ── Verification mode (default) ──────────────────────────────────────
    std::cout << "=== Forward-pass scores (align_combo) ===\n";
    auto scores_ref = align_combo(
        reads, vertex_labels, inGraph,
        origin_string_graph, contracted_str_to_og,
        match_score, subst_score, del_score, ins_score,
        query, mapper_str);

    for (size_t i = 0; i < reads.size(); ++i)
        std::cout << "read[" << i << "] score=" << scores_ref[i] << "\n";

    std::cout << "\n=== Traceback results (align_combo_with_traceback) ===\n";
    std::cout << "  (checkpoint every k=" << cfg.k << " rows, dir=" << cfg.dir << ")\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto alignments = align_combo_with_traceback(
        reads, vertex_labels, inGraph,
        origin_string_graph, contracted_str_to_og,
        match_score, subst_score, del_score, ins_score,
        query, mapper_str, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool all_ok = true;
    for (size_t ri = 0; ri < reads.size(); ++ri) {
        const auto& path  = alignments[ri];
        const auto& read  = reads[ri];
        std::string cigar = ops_to_cigar(path);

        int path_score = 0;
        int read_pos = -1;
        for (const auto& op : path) {
            switch (op.op) {
                case BP_DELETE:
                    path_score += del_score;
                    read_pos = static_cast<int>(op.row);
                    break;
                case BP_MATCHSUB:
                    path_score += (vertex_labels[op.vertex] == read[op.row]) ? match_score : subst_score;
                    read_pos = static_cast<int>(op.row);
                    break;
                case BP_INSERT:
                    path_score += ins_score;
                    break;
            }
        }

        bool score_ok  = (path_score == scores_ref[ri]);
        bool len_ok    = (read_pos == static_cast<int>(read.size()) - 1);

        std::cout << "read[" << ri << "] = \"" << read << "\"\n"
                  << "  CIGAR      : " << cigar << "\n"
                  << "  path_score : " << path_score
                  << "  ref_score  : " << scores_ref[ri]
                  << "  score_ok   : " << (score_ok  ? "YES" : "FAIL")  << "\n"
                  << "  last_row   : " << read_pos
                  << "  expected   : " << (read.size() - 1)
                  << "  row_ok     : " << (len_ok ? "YES" : "FAIL") << "\n";

        if (!score_ok || !len_ok) all_ok = false;
    }

    std::cout << "\n[traceback time] " << ms << " ms\n";
    std::cout << "\n[RESULT] " << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << "\n";
    return all_ok ? 0 : 1;
}

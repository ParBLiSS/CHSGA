// ParSGA_traceback.cpp
// Same inputs as ParSGA.cpp, plus optional checkpoint flags:
//   [-b]         benchmark mode: time score+traceback only
//   [-k rows] [-d ckpt_dir]
//
// Positional arguments (same order as ParSGA):
//   <csr_prefix> <output_file_unused> <reads.fastq> <num_threads>

#include <getopt.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "csr_traceback.hpp"
#include "read_csr.hpp"
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
    ParCheckpointConfig cfg;
    cfg.k       = 500;
    cfg.dir     = "/tmp/parsga_traceback_ckpts";
    cfg.cleanup = true;
    bool benchmark = false;

    int opt;
    while ((opt = getopt(argc, argv, "bk:d:")) != -1) {
        switch (opt) {
            case 'b': benchmark = true; break;
            case 'k': cfg.k = std::stoll(optarg); break;
            case 'd': cfg.dir = optarg; break;
            default:
                std::cerr
                    << "Usage: " << argv[0]
                    << " [-b] [-k checkpoint_every_k_rows] [-d ckpt_dir]\n"
                    << "       <csr_prefix> <output_unused> <reads> <num_threads>\n";
                return EXIT_FAILURE;
        }
    }

    if (argc - optind < 4) {
        std::cerr << "Error: need <csr_prefix> <output> <reads> <num_threads>\n";
        return EXIT_FAILURE;
    }

    const char* input_prefix   = argv[optind];
    const char* reads_filename = argv[optind + 2];
    const int   num_threads    = std::stoi(argv[optind + 3]);

    int const max_threads = omp_get_max_threads();
    if (num_threads > max_threads) {
        std::cerr << "Requested " << num_threads << " threads but max is "
                  << max_threads << '\n';
        return EXIT_FAILURE;
    }
    omp_set_num_threads(num_threads);
    std::cout << "set threads: " << omp_get_max_threads() << '\n' << std::flush;

    int match_score = 0, subst_score = 1, del_score = 1, ins_score = 1;
    std::cout << std::fixed << std::setprecision(6);

    int n = 0, m = 0;
    std::vector<char> labels;
    std::vector<int> offsets, edges, cut_offsets_out[ALPHABET_SIZE],
        cut_adjcny_out[ALPHABET_SIZE], component_indices[ALPHABET_SIZE],
        component_labels[ALPHABET_SIZE], thread_component_offsets[ALPHABET_SIZE];

    std::vector<std::string> reads = load_reads(reads_filename);

    double input_start = omp_get_wtime();
    parallel_input(
        std::string(input_prefix), n, m, labels, offsets, edges,
        cut_offsets_out, cut_adjcny_out, component_indices, component_labels,
        thread_component_offsets);
    std::cout << "input time: " << (omp_get_wtime() - input_start) << '\n'
              << std::flush;

    if (benchmark) {
        std::vector<double> per_read_sec;
        per_read_sec.reserve(reads.size());

        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t ri = 0; ri < reads.size(); ++ri) {
            ParCheckpointConfig rcfg = cfg;
            rcfg.dir = cfg.dir + "/read_" + std::to_string(ri);

            auto r0 = std::chrono::high_resolution_clock::now();
            (void) par_csr_align_read_with_traceback<uint8_t>(
                reads[ri], n, labels, offsets, edges,
                cut_offsets_out, cut_adjcny_out,
                component_indices, thread_component_offsets,
                match_score, subst_score, del_score, ins_score, rcfg);
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

    std::vector<double> op_times_serial[NUM_OPERATIONS];
    std::vector<double> op_times_parallel[NUM_OPERATIONS];

    auto scores_serial = serial_align_CSR<uint8_t>(
        reads, n, labels, offsets, edges, match_score, subst_score, del_score,
        ins_score, op_times_serial);

    auto scores_parallel = parallel_align_CSR<uint8_t>(
        reads, n, labels, offsets, edges, cut_offsets_out, cut_adjcny_out,
        component_indices, thread_component_offsets, match_score, subst_score,
        del_score, ins_score, op_times_parallel);

    std::cout << "\n=== Forward-pass scores (parallel_align_CSR — traceback reference) ===\n";
    for (size_t i = 0; i < reads.size(); ++i)
        std::cout << "read[" << i << "] parallel_score="
                  << static_cast<int>(scores_parallel[i]) << '\n';

    std::cout << "\n=== serial_align_CSR scores (informational) ===\n";
    for (size_t i = 0; i < reads.size(); ++i)
        std::cout << "read[" << i << "] serial_score="
                  << static_cast<int>(scores_serial[i]) << '\n';

    std::cout << "\n=== Traceback (checkpoint every k=" << cfg.k
              << " rows, dir=" << cfg.dir << ") ===\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto alignments = par_csr_align_reads_with_traceback<uint8_t>(
        reads, n, labels, offsets, edges,
        cut_offsets_out, cut_adjcny_out,
        component_indices, thread_component_offsets,
        match_score, subst_score, del_score, ins_score, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool all_ok = true;
    for (size_t ri = 0; ri < reads.size(); ++ri) {
        const auto& path = alignments[ri];
        const auto& read = reads[ri];
        const std::string cigar = par_ops_to_cigar(path);

        int path_score = 0;
        int read_pos   = -1;
        for (const auto& op : path) {
            switch (op.op) {
                case ParBP_DELETE:
                    path_score += del_score;
                    read_pos = static_cast<int>(op.row);
                    break;
                case ParBP_MATCHSUB:
                    path_score += (labels[static_cast<size_t>(op.vertex)] == read[static_cast<size_t>(op.row)])
                                        ? match_score
                                        : subst_score;
                    read_pos = static_cast<int>(op.row);
                    break;
                case ParBP_INSERT:
                    path_score += ins_score;
                    break;
            }
        }

        const bool score_ok =
            (path_score == static_cast<int>(scores_parallel[ri]));
        const bool len_ok =
            (read_pos == static_cast<int>(read.size()) - 1);

        std::cout << "read[" << ri << "] = \"" << read << "\"\n"
                  << "  CIGAR       : " << cigar << '\n'
                  << "  path_score  : " << path_score                   << "  parallel_ref : "
                  << static_cast<int>(scores_parallel[ri]) << "  score_ok : "
                  << (score_ok ? "YES" : "FAIL") << '\n'
                  << "  last_row    : " << read_pos << "  expected : "
                  << (read.size() - 1) << "  row_ok : "
                  << (len_ok ? "YES" : "FAIL") << '\n';

        if (!score_ok || !len_ok) all_ok = false;
    }

    std::cout << "\n[traceback time] " << ms << " ms\n";
    std::cout << "\n[RESULT] "
              << (all_ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED") << '\n';
    return all_ok ? 0 : 1;
}

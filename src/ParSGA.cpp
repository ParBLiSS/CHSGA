#include <iostream>
#include <iomanip>
#include "csr_align.hpp"
#include "read_csr.hpp"
#include "load_reads.hpp"

template<typename... Args>
void print_variables(std::ofstream& out, Args&&... args) {
    std::ostringstream oss;
    ((oss << std::forward<Args>(args) << ","), ...);
    oss << "\n";
    out << oss.str();
}


template <typename T>
bool areVectorsEqual(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    return (vec1.size() == vec2.size() && std::equal(vec1.begin(), vec1.end(), vec2.begin()));
}

int main(int argc, char **argv) {
    if (argc < 5) {
        throw std::runtime_error("Provided less than 4 arguments");
    }

    int match_score, subst_score, del_score, ins_score;
    match_score = 0;
    subst_score = 1;
    del_score = 1;
    ins_score = 1;
    // print with single-point precision
    std::cout << std::fixed << std::setprecision(6);
    double input_time = 0;

    int n, m;
    std::vector<char> labels;
    std::vector<int> offsets, edges, cut_offsets_out[ALPHABET_SIZE], cut_adjcny_out[ALPHABET_SIZE], component_indices[ALPHABET_SIZE], component_labels[ALPHABET_SIZE], thread_component_offsets[ALPHABET_SIZE];

    const char* reads_filename = argv[3];
    std::vector<std::string> reads = load_reads(reads_filename);

    int const max_threads = omp_get_max_threads(); // remember what max threads are before setting num threads to 1
    int const num_threads = std::stoi(argv[4]);
    if (num_threads > max_threads) {
        throw std::runtime_error("Requested " + std::to_string(num_threads) + " on machine with " + std::to_string(max_threads) + " cores\n");
    }
    omp_set_num_threads(num_threads);
    std::cout << "set threads: " << omp_get_max_threads() << '\n' << std::flush;

    double input_start_time = omp_get_wtime();
    parallel_input(std::string(argv[1]), n, m, labels, offsets, edges, cut_offsets_out, cut_adjcny_out, component_indices, component_labels, thread_component_offsets);
    input_time = omp_get_wtime() - input_start_time;
    std::cout << "input time: " << input_time << '\n' << std::flush;
    std::string output_filename(argv[2]);
    std::cout<<"\n";

    // Per-operation, per-read timing vectors (one vector per operation)
    std::vector<double> op_times[NUM_OPERATIONS];

    // Call the updated parallel_align_CSR which appends per-read times into op_times
    std::vector<uint8_t> parallel_score = parallel_align_CSR<uint8_t>(
        reads,
        n,
        labels,
        offsets,
        edges,
        cut_offsets_out,
        cut_adjcny_out,
        component_indices,
        thread_component_offsets,
        match_score,
        subst_score,
        del_score,
        ins_score,
        op_times
    );

    // Print timing summary using the same style as ConSGA's align_combo (skip first read when computing averages)
    std::cout << "Average times (skipping first read), 95% CI:\n";
    for (int op_num = 0; op_num < NUM_OPERATIONS; ++op_num) {
        std::string label = " " + operationToString(static_cast<Operation>(op_num));
        print_mean_with_moe(label.c_str(), op_times[op_num]);
    }
    printVector("score", parallel_score);
    return 0;
}
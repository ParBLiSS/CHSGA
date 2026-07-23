#include <iostream>
#include <iomanip>
#include <numeric> // for accumulate
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

double imbalance(double const &max_element, std::vector<double>& elements) {
    return max_element / (std::accumulate(elements.begin(), elements.end(), 0.0) / elements.size());
}

int main(int argc, char **argv) {
    if (argc < 4) {
        throw std::runtime_error("Provided less than 3 arguments");
    }
    int match_score, subst_score, del_score, ins_score;
    match_score = 0;
    subst_score = 1;
    del_score = 1;
    ins_score = 1;
    // print with single-point precision
    std::cout << std::fixed << std::setprecision(6);
    double input_time = 0, serial_time = 0;

    int n = 0;
    int m = 0;
    std::vector<char> labels;
    std::vector<int> offsets;
    std::vector<int> edges;
    
    const char* reads_filename = argv[3];
    std::vector<std::string> reads = load_reads(reads_filename);
    

    double input_start_time = omp_get_wtime();
    serial_input(std::string(argv[1]), n, m, labels, offsets, edges);
    input_time = omp_get_wtime() - input_start_time;
    std::cout << "input time: " << input_time << std::endl;
    std::string output_filename(argv[2]);

    double serial_start_time = omp_get_wtime();

    // Per-operation, per-read timing vectors (one vector per operation)
    std::vector<double> op_times[NUM_OPERATIONS];
    // Call the updated serial_align_CSR which appends per-read times into op_times
    std::vector<uint8_t> serial_score = serial_align_CSR<uint8_t>(
        reads,
        n,
        labels,
        offsets,
        edges,
        match_score,
        subst_score,
        del_score,
        ins_score,
        op_times
    );

    serial_time = omp_get_wtime() - serial_start_time;
    std::cout << "serial time: " << serial_time << std::endl;

    // Print timing summary using the same style as ConSGA's align_combo (skip first read when computing averages)
    std::cout << "Average times (skipping first read), 95% CI:\n";
    for (int op_num = 0; op_num < NUM_OPERATIONS; ++op_num) {
        std::string label = " " + operationToString(static_cast<Operation>(op_num));
        print_mean_with_moe(label.c_str(), op_times[op_num]);
    }
    printVector("score", serial_score);
    return 0;
}
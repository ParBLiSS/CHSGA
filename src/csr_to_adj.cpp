// csr2adj.cpp
// Reads a text CSR (space-separated labels) and writes the AdjacencyGraph
// textual format used by your pipeline. Keeps the adjacency output layout
// unchanged but adds input validation and clearer errors.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: csr2adj IN.csr OUT.adj OUT.labels OUT.label_offsets\n";
        return 2;
    }

    std::ifstream in(argv[1]);
    std::ofstream out_adj(argv[2]);
    std::ofstream out_labels(argv[3]);
    std::ofstream out_label_offsets(argv[4]);
    if (!in) {
        std::cerr << "File open error: cannot open input CSR file: " << argv[1] << "\n";
        return 3;
    }
    if (!out_adj) {
        std::cerr << "File open error: cannot open output adjacency file: " << argv[2] << "\n";
        return 3;
    }
    if (!out_labels) {
        std::cerr << "File open error: cannot open output labels file: " << argv[3] << "\n";
        return 3;
    }
    if (!out_label_offsets) {
        std::cerr << "File open error: cannot open output label offsets file: " << argv[4] << "\n";
        return 3;
    }

    std::uint64_t n = 0, m = 0;
    if (!(in >> n >> m)) {
        std::cerr << "Parse error: failed to read header (n m) from CSR\n";
        return 4;
    }

    // Write adjacency header
    out_adj << "AdjacencyGraph\n" << n << '\n' << m << '\n';

    // Read labels (space-separated). This assumes labels contain no whitespace.
    std::string lab;
    std::uint64_t cur = 0;
    std::vector<std::string> labels;
    labels.reserve(n);
    for (std::uint64_t i = 0; i < n; ++i) {
        if (!(in >> lab)) {
            std::cerr << "Parse error: failed to read label " << i << " of " << n << "\n";
            return 5;
        }
        labels.push_back(lab);
        out_labels << lab;
        out_label_offsets << cur << ' ';
        cur += lab.size();
    }
    out_label_offsets << cur << '\n';

    // Read offsets: expect n+1 entries
    std::vector<std::uint64_t> offsets;
    offsets.reserve(n + 1);
    std::uint64_t off = 0;
    for (std::uint64_t i = 0; i < n + 1; ++i) {
        if (!(in >> off)) {
            std::cerr << "Parse error: failed to read offset " << i << " of " << (n + 1) << "\n";
            return 6;
        }
        offsets.push_back(off);
        // The original pipeline writes only first n offsets to out_adj; preserve that behavior.
        if (i < n) out_adj << off << '\n';
    }

    // Validate monotonicity of offsets and that total edges equals m
    if (!offsets.empty()) {
        for (std::size_t i = 1; i < offsets.size(); ++i) {
            if (offsets[i] < offsets[i - 1]) {
                std::cerr << "Parse error: offsets not non-decreasing at index " << i << "\n";
                return 6;
            }
        }
        if (offsets.back() != m) {
            std::cerr << "Warning: final offset (" << offsets.back()
                      << ") does not equal declared edge count m (" << m << ")\n";
            // treat as non-fatal but prefer to continue; you can change to fatal if desired
        }
    }

    // Read adjacency list: expect m entries
    std::vector<std::uint64_t> adj;
    adj.reserve(m);
    std::uint64_t dst = 0;
    for (std::uint64_t i = 0; i < m; ++i) {
        if (!(in >> dst)) {
            std::cerr << "Parse error: failed to read adjacency entry " << i << " of " << m << "\n";
            return 7;
        }
        adj.push_back(dst);
        out_adj << dst << '\n';
    }

    if (adj.size() != m) {
        std::cerr << "Parse error: adjacency count mismatch: read " << adj.size()
                  << " expected " << m << "\n";
        return 7;
    }

    // Write default weights (1.0) for each edge, as original code did
    for (std::uint64_t i = 0; i < m; ++i) out_adj << "1.0\n";

    // Final sanity checks (optional)
    if (offsets.size() != n + 1) {
        std::cerr << "Final check failed: offsets size " << offsets.size()
                  << " != " << (n + 1) << "\n";
        return 8;
    }

    return 0;
}

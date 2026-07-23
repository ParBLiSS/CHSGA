#include <zlib.h>
#include "kseq.h"
KSEQ_INIT(gzFile, gzread);

constexpr size_t MAX_READS_TO_LOAD = 5;

std::vector<std::string> load_reads(const char* reads_filename) {
    std::vector<std::string> reads;

    gzFile fp = gzopen(reads_filename, "r");
    if (!fp) {
        throw std::runtime_error(std::string("Cannot open reads file: ") + reads_filename);
    }

    kseq_t* seq = kseq_init(fp);
    int len;
    size_t loaded = 0;

    while ((len = kseq_read(seq)) >= 0) {
        // Convert sequence to uppercase
        std::string s(seq->seq.s, seq->seq.l);
        for (char& c : s) {
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        reads.push_back(std::move(s));

        ++loaded;
        if (MAX_READS_TO_LOAD != 0 && loaded >= MAX_READS_TO_LOAD) {
            break;
        }
    }

    kseq_destroy(seq);
    gzclose(fp);

    if (reads.empty()) {
        throw std::runtime_error(std::string("No reads loaded from: ") + reads_filename);
    }

    return reads;
}
// handlegraph_to_csr.cpp
// Fixed version of the program that loads a PackedGraph, converts it to a
// CSR string-labeled graph and a character-labeled CSR graph, and writes both.
//
// Build: link with libbdsg / libhandlegraph as appropriate for your environment.

#include <bdsg/packed_graph.hpp>
#include <handlegraph/handle_graph.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace bdsg;
using handlegraph::HandleGraph;

struct CSR_container {
    uint64_t numVertices = 0;
    uint64_t numEdges    = 0;

    std::vector<std::string> vertex_metadata;
    std::vector<uint64_t> offsets_out;
    std::vector<uint64_t> adjcny_out;

    void clear() {
        numVertices     = 0;
        numEdges        = 0;
        vertex_metadata.clear();
        offsets_out.clear();
        adjcny_out.clear();
    }

    void reserve(uint64_t nV, uint64_t nE) {
        vertex_metadata.reserve(nV);
        offsets_out.reserve(nV + 1);
        adjcny_out.reserve(nE);
    }
};

void fill_csr_from_handlegraph(const HandleGraph& graph,
                               CSR_container& csr) {

    csr.clear();

    // 1) Gather forward handles and sort by node ID
    std::vector<handle_t> fwd_handles;
    fwd_handles.reserve(graph.get_node_count());
    graph.for_each_handle([&](const handle_t& h) {
        fwd_handles.push_back(h);
        return true;
    });
    std::sort(fwd_handles.begin(), fwd_handles.end(),
              [&](const handle_t& a, const handle_t& b) {
                  return graph.get_id(a) < graph.get_id(b);
              });

    // 2) Build oriented handle list: forward then flipped for each node
    std::vector<handle_t> handles;
    handles.reserve(2 * fwd_handles.size());
    for (auto& h : fwd_handles) {
        handles.push_back(h);
        handles.push_back(graph.flip(h));
    }

    // 3) Index map: (node_id<<1 | rev_bit) -> CSR vertex index
    std::unordered_map<uint64_t, uint64_t> handle_index;
    handle_index.reserve(handles.size());
    for (uint64_t i = 0; i < handles.size(); ++i) {
        const auto& h = handles[i];
        uint64_t id = graph.get_id(h);
        uint64_t rev_bit = graph.get_is_reverse(h) ? 1ULL : 0ULL;
        uint64_t key = (id << 1) | rev_bit;
        handle_index[key] = i;
    }

    // 4) Populate vertex metadata
    csr.numVertices = handles.size();
    csr.vertex_metadata.resize(csr.numVertices);
    for (uint64_t i = 0; i < csr.numVertices; ++i) {
        csr.vertex_metadata[i] = graph.get_sequence(handles[i]);
    }

    // 5) Build CSR for directed outgoing arcs
    csr.offsets_out.clear();
    csr.adjcny_out.clear();
    csr.offsets_out.reserve(csr.numVertices + 1);
    csr.offsets_out.push_back(0);

    for (uint64_t i = 0; i < csr.numVertices; ++i) {
        const handle_t& h = handles[i];
        uint64_t out_count = 0;

        graph.follow_edges(h, /* go_left = */ false,
                           [&](const handle_t& to) {
            uint64_t to_key = (graph.get_id(to) << 1)
                              | (graph.get_is_reverse(to) ? 1ULL : 0ULL);
            uint64_t to_idx = handle_index.at(to_key);
            csr.adjcny_out.push_back(to_idx);
            ++out_count;
            return true;
        });

        csr.offsets_out.push_back(csr.offsets_out.back() + out_count);
    }

    // 6) Finalize counts and ensure offsets are valid
    csr.numEdges = csr.adjcny_out.size();

    if (csr.offsets_out.size() != csr.numVertices + 1) {
        std::cerr << "fill_csr_from_handlegraph(): unexpected offsets_out size\n";
        std::exit(1);
    }

    // Sort adjacency ranges per-vertex for deterministic order
    for (uint64_t v = 0; v < csr.numVertices; ++v) {
        auto b = csr.offsets_out[v];
        auto e = csr.offsets_out[v + 1];
        if (e > b) {
            std::sort(csr.adjcny_out.begin() + b,
                      csr.adjcny_out.begin() + e);
        }
    }
}

void write_csr(const CSR_container& csr, const std::string& filename) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening output file: " << filename << "\n";
        std::exit(1);
    }

    // Header: number of vertices, number of edges
    out << csr.numVertices << "\n"
        << csr.numEdges    << "\n";

    // Vertex labels (space-separated). This assumes labels contain no spaces.
    for (size_t v = 0; v < csr.numVertices; ++v) {
        out << csr.vertex_metadata[v] << " ";
    }
    out << "\n";

    // CSR offsets (n+1 entries)
    for (size_t i = 0; i < csr.offsets_out.size(); ++i) {
        out << csr.offsets_out[i] << " ";
    }
    out << "\n";

    // Adjacency list (outgoing edges)
    for (size_t e = 0; e < csr.adjcny_out.size(); ++e) {
        out << csr.adjcny_out[e] << " ";
    }
    out << "\n";
}

/**
 * Convert a string-labeled CSR (src) into a character-labeled CSR (dst).
 * Each character becomes a vertex. Internal contiguity edges connect chars
 * within the same original string; the last char of each string inherits the
 * original outgoing edges (mapped to the first char of the neighbor string).
 */
void build(const CSR_container& src, CSR_container& dst)
{
    const uint64_t N = src.numVertices;
    std::vector<uint64_t> charPrefix(N + 1, 0);
    for (uint64_t i = 0; i < N; ++i) {
        charPrefix[i + 1] = charPrefix[i] + src.vertex_metadata[i].size();
    }

    const uint64_t totalChars = charPrefix[N];

    // Count edges: internal contiguity edges + external edges
    const uint64_t internalEdges = totalChars > N ? (totalChars - N) : 0;
    const uint64_t externalEdges = src.adjcny_out.size();
    const uint64_t totalEdges    = internalEdges + externalEdges;

    dst.clear();
    dst.numVertices = totalChars;
    // don't trust this early; we'll set the authoritative numEdges after build
    dst.reserve(totalChars, totalEdges);

    // Build character-level vertex metadata
    dst.vertex_metadata.resize(totalChars);
    for (uint64_t i = 0; i < N; ++i) {
        const std::string& label = src.vertex_metadata[i];
        uint64_t base = charPrefix[i];
        for (uint64_t j = 0; j < label.size(); ++j) {
            dst.vertex_metadata[base + j] = std::string(1, label[j]);
        }
    }

    // Build offsets_out and adjcny_out
    dst.offsets_out.clear();
    dst.adjcny_out.clear();
    dst.offsets_out.reserve(totalChars + 1);
    dst.adjcny_out.reserve(totalEdges);

    uint64_t edgeCount = 0;
    dst.offsets_out.push_back(0);

    // For each original string i
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t start = charPrefix[i];
        uint64_t end   = charPrefix[i + 1];

        // Handle empty-label vertices: still must emit an offsets entry
        if (start == end) {
            // No character vertices for this string; push one offsets entry
            dst.offsets_out.push_back(edgeCount);
            continue;
        }

        // For each character vertex v in this string
        for (uint64_t v = start; v < end; ++v) {
            // contiguity edge to next character if not at end
            if (v + 1 < end) {
                dst.adjcny_out.push_back(v + 1);
                ++edgeCount;
            } else {
                // last character: inherit external outgoing edges of string i
                uint64_t oBeg = src.offsets_out[i];
                uint64_t oEnd = src.offsets_out[i + 1];
                for (uint64_t e = oBeg; e < oEnd; ++e) {
                    uint64_t nei = src.adjcny_out[e];
                    // map neighbor-string ID -> its first character
                    uint64_t first_char_of_nei = charPrefix[nei];
                    dst.adjcny_out.push_back(first_char_of_nei);
                    ++edgeCount;
                }
            }
            dst.offsets_out.push_back(edgeCount);
        }
    }

    // Authoritative counts and sanity checks
    dst.numEdges = dst.adjcny_out.size();
    if (dst.offsets_out.size() != dst.numVertices + 1) {
        // If some source vertices were empty and caused mismatch, we attempted
        // to handle them above; if we still fail, that's an error.
        std::cerr << "build(): offsets_out size mismatch: "
                  << dst.offsets_out.size() << " vs " << (dst.numVertices + 1) << "\n";
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    // Program expects three arguments:
    //   1) input packed graph (serialized PackedGraph)
    //   2) output CSR file for string-labeled graph
    //   3) output CSR file for character-labeled graph
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <packed_in> <packed_string_csr_out> <packed_char_csr_out>\n";
        return 1;
    }

    const std::string packedIn   = argv[1];
    const std::string packedStringOut = argv[2];
    const std::string packedCharOut   = argv[3];

    PackedGraph packed_graph;

    // PackedGraph::deserialize may accept an istream depending on bdsg version.
    // Use an ifstream to be compatible with versions that require a stream.
    std::ifstream pin(packedIn, std::ios::binary);
    if (!pin) {
        std::cerr << "Error opening packed graph file: " << packedIn << "\n";
        return 1;
    }
    // If your bdsg version provides deserialize(const std::string&), you can
    // alternatively call packed_graph.deserialize(packedIn).
    try {
        packed_graph.deserialize(pin);
    } catch (const std::exception& e) {
        std::cerr << "PackedGraph deserialize failed: " << e.what() << "\n";
        return 1;
    }

    CSR_container csr_packed, csr_char_packed;
    fill_csr_from_handlegraph(packed_graph, csr_packed);
    build(csr_packed, csr_char_packed);

    write_csr(csr_packed,      packedStringOut);
    write_csr(csr_char_packed, packedCharOut);

    return 0;
}

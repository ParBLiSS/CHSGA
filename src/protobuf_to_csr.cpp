#include <iostream>
#include <iomanip>
#include "graphLoad.hpp"

int main(int argc, char **argv) {
    if (argc < 4) {
        throw std::runtime_error("Provided less than 3 arguments");
    }
    bool match_cut;
    int match_score, subst_score, del_score, ins_score;

    psgl::graphLoader g;
    std::string graph_filename = argv[1];
    if(graph_filename.substr(graph_filename.length()-3)==".vg") {
        g.loadFromVG(graph_filename);
    } else if (graph_filename.substr(graph_filename.length()-4)==".txt") {
        g.loadFromTxt(graph_filename);
    } else {
        std::cerr << "error: graph filename should end with .vg or .txt\n" <<std::endl;
    };

    // 1) CHAR‐labeled graph
    {
        std::ofstream out(argv[3]);
        if (!out.is_open()) {
            std::cerr << "Error opening char file: " << argv[3] << "\n";
            return 1;
        } else {
            std::cout << "Opened char file" << argv[3] << "\n";
        }

        // Line 1: numVertices numEdges
        out << g.diCharGraph.numVertices
            << " " << g.diCharGraph.numEdges << "\n";

        // Line 2: write each label as a whitespace-separated token (one token per vertex)
        for (int i = 0; i < g.diCharGraph.numVertices; ++i) {
            out << g.diCharGraph.vertex_label[i];
            if (i + 1 < g.diCharGraph.numVertices) out << " ";
        }
        out << "\n";
        // Line 3: offsets_out (size = numVertices+1)
        for (auto ofs : g.diCharGraph.offsets_out)
            out << ofs << " ";
        out << "\n";

        // Line 4: adjcny_out (size = numEdges)
        for (auto e : g.diCharGraph.adjcny_out)
            out << e << " ";
        out << "\n";

        // Line 5: offsets_in (new; size = numVertices+1)
        for (auto ofs : g.diCharGraph.offsets_in)
            out << ofs << " ";
        out << "\n";

        // Line 6: adjcny_in (new; size = numEdges)
        for (auto e : g.diCharGraph.adjcny_in)
            out << e << " ";
        out << "\n";
    }

    // 2) STRING‐labeled graph
    {
        std::ofstream out(argv[2]);
        if (!out.is_open()) {
            std::cerr << "Error opening string file: " << argv[2] << "\n";
            return 1;
        } else {
            std::cout << "Opened string file" << argv[2] << "\n";
        }

        out << g.diGraph.numVertices
            << " " << g.diGraph.numEdges << "\n";

        // Whitespace‐separated strings
        for (int i = 0; i < g.diGraph.numVertices; ++i)
            out << g.diGraph.vertex_metadata[i] << " ";
        out << "\n";

        for (auto ofs : g.diGraph.offsets_out)
            out << ofs << " ";
        out << "\n";

        for (auto e : g.diGraph.adjcny_out)
            out << e << " ";
        out << "\n";

        for (auto ofs : g.diGraph.offsets_in)
            out << ofs << " ";
        out << "\n";

        for (auto e : g.diGraph.adjcny_in)
            out << e << " ";
        out << "\n";
    }

    return 0;
}

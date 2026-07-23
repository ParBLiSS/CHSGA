#include <fstream>  // for std::ifstream

void serial_input(std::string input_prefix, int &n, int &m, std::vector<char> &labels, std::vector<int> &offsets, std::vector<int> &edges) {
    std::string csr_filename(input_prefix+".csr");
    std::ifstream csr_fin(csr_filename);
    if (csr_fin.is_open()) {
        csr_fin >> n;
        csr_fin >> m;
        labels.resize(n);
        for (int i = 0; i < n; ++i) {
            csr_fin >> labels[i];
        }
        offsets.resize(n+1);
        for (int i = 0; i < n+1; ++i) {
            csr_fin >> offsets[i];
        }
        edges.resize(m);
        for (int i = 0; i < m; ++i) {
            csr_fin >> edges[i];
        }
        csr_fin.close();
        std::cout << "Read csr file: "+ csr_filename << '\n';
        std::cout << "n: " << n << '\n';
        std::cout << "m: " << m << '\n';
        std::cout << "labels size: " << labels.size() << '\n';
        std::cout << "offsets size: " << offsets.size() << '\n';
        std::cout << "edges size: " << edges.size() << '\n';
    } else {
        throw std::runtime_error("Unable to open csr file: "+ csr_filename);
    }
}

// separated for executables that change thread offsets
void thread_offsets_input(std::string input_prefix, std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE]) {
    for(char base: bases) {
        std::string thread_offsets_filename(input_prefix+"-"+base+"-"+std::to_string(omp_get_max_threads())+".components");
        std::ifstream thread_offset_fin(thread_offsets_filename);
        if (thread_offset_fin.is_open()) {
            for (int i = 0; i < omp_get_max_threads()+1; ++i) {
                thread_offset_fin >> thread_component_offsets[base_index(base)][i];
            }
            thread_offset_fin.close();
            std::cout << "Read thread offsets file: "+ thread_offsets_filename << '\n';
        } else {
            throw std::runtime_error("Unable to open thread offsets file: "+ thread_offsets_filename);
        }
    }
}

void parallel_input(std::string input_prefix, int &n, int &m, std::vector<char> &labels, std::vector<int> &offsets, std::vector<int> &edges, std::vector<int> (&cut_offsets_out)[ALPHABET_SIZE], std::vector<int> (&cut_adjcny_out)[ALPHABET_SIZE], std::vector<int> (&component_indices)[ALPHABET_SIZE], std::vector<int> (&component_labels)[ALPHABET_SIZE], std::vector<int> (&thread_component_offsets)[ALPHABET_SIZE]) {
    serial_input(input_prefix, n, m, labels, offsets, edges);
    double parallel_input_start_time = omp_get_wtime();
    for(char base: bases) {
        std::string cut_filename(input_prefix+"-"+base+".csr");
        std::ifstream cut_fin(cut_filename);
        int cut_n, cut_m;
        if (cut_fin.is_open()) {
            cut_fin >> cut_n;
            if(cut_n != n) {
                throw std::runtime_error("Cut file: "+ cut_filename + " has wrong number of vertices: " + std::to_string(cut_n));
            }
            cut_fin >> cut_m;
            cut_offsets_out[base_index(base)].resize(cut_n+1);
            cut_adjcny_out[base_index(base)].resize(cut_m);
            component_indices[base_index(base)].resize(cut_n);
            component_labels[base_index(base)].resize(cut_n);
            thread_component_offsets[base_index(base)].resize(omp_get_max_threads()+1);
            for (int i = 0; i < cut_n+1; ++i) {
                cut_fin >> cut_offsets_out[base_index(base)][i];
            }
            for (int i = 0; i < cut_m; ++i) {
                cut_fin >> cut_adjcny_out[base_index(base)][i];
            }
            cut_fin.close();
            std::cout << "Read cut file: "+ cut_filename << '\n';
        } else {
            throw std::runtime_error("Unable to open cut file: "+ cut_filename);
        }
        std::string components_filename(input_prefix+"-"+base+".components");
        std::ifstream component_fin(components_filename);
        if (component_fin.is_open()) {
            for (int i = 0; i < cut_n; ++i) {
                component_fin >> component_indices[base_index(base)][i];
            }
            for (int i = 0; i < cut_n; ++i) {
                component_fin >> component_labels[base_index(base)][i];
            }
            component_fin.close();
            std::cout << "Read cut components file: "+ components_filename << '\n';
        } else {
            throw std::runtime_error("Unable to open cut components file: "+ components_filename);
        }
    }
    thread_offsets_input(input_prefix, thread_component_offsets);
    double parallel_input_time = omp_get_wtime() - parallel_input_start_time;
    std::cout << "parallel input time: " << parallel_input_time << '\n';
}

CXX ?= g++
CXXFLAGS ?= -O3 -g -I src/

CON_INCLUDES = -I src/ParSGAoverrides/ -I ext/PaSGAL/ext/kseq/ -lz \
	-I ext/Parallel-Contraction-Hierarchy/ \
	-I ext/Parallel-Contraction-Hierarchy/parlaylib/include/ \
	-I ext/ParSGA/src
PAR_INCLUDES = -I src/ParSGAoverrides/ -I ext/PaSGAL/ext/kseq/ -lz

# Optional: set to "-DUSE_PAPI -lpapi" to enable PAPI counters in ConSGA.
PAPI_FLAGS ?=

# Raised in-memory score/backpointer store for very large graphs (e.g. Opossum).
# Default fast-path cap in the sources is 400 GiB.
FAST_STORE_800GB = -DFAST_BP_MAX_BYTES_OVERRIDE=858993459200

.PHONY: clean all all_gen_graph all_dag all_traceback_dag

clean:
	rm -rf handle_build handle_unidirectedbuild handlegraph_to_csr \
		handlegraph_to_unidirectedcsr protobuf_to_csr_build protobuf_to_csr \
		csr_to_adj build_pch \
		char_graph_build_matchN csr_to_char_graphs_matchN \
		char_graph_build_mismatchN csr_to_char_graphs_mismatchN \
		serialParSGA_matchN serialParSGA_mismatchN \
		ParSGA_matchN ParSGA_mismatchN ConSGA_matchN ConSGA_mismatchN \
		serialParSGA_matchN_dag serialParSGA_mismatchN_dag \
		ParSGA_matchN_dag ParSGA_mismatchN_dag \
		ConSGA_matchN_dag ConSGA_mismatchN_dag \
		ConSGA_traceback_matchN ConSGA_traceback_mismatchN \
		ConSGA_traceback_matchN_dag ConSGA_traceback_mismatchN_dag \
		ParSGA_traceback_matchN_dag ParSGA_traceback_mismatchN_dag \
		ConSGA_traceback_mismatchN_dag_800gb \
		ParSGA_traceback_mismatchN_dag_800gb \
		PaSGAL PaSGAL_match ConSGA_mismatchNpapi

all: handlegraph_to_csr protobuf_to_csr csr_to_adj build_pch \
	csr_to_char_graphs_matchN csr_to_char_graphs_mismatchN \
	serialParSGA_matchN serialParSGA_mismatchN \
	ParSGA_matchN ParSGA_mismatchN ConSGA_matchN ConSGA_mismatchN \
	serialParSGA_matchN_dag serialParSGA_mismatchN_dag \
	ParSGA_matchN_dag ParSGA_mismatchN_dag \
	ConSGA_matchN_dag ConSGA_mismatchN_dag \
	PaSGAL PaSGAL_match

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Requires a conda environment with libbdsg build dependencies available.
handlegraph_to_csr:
	@mkdir -p handle_build
	@bash -c 'export PKG_CONFIG_PATH=$${CONDA_PREFIX:+$$CONDA_PREFIX/lib/pkgconfig:}$$PKG_CONFIG_PATH && \
		cd handle_build && CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" \
		cmake .. \
			-DBUILD_HANDLEGRAPH_TO_CSR=ON -DBUILD_CSR_TO_CHAR_GRAPHS=OFF -DBUILD_PROTOBUF_TO_CSR=OFF \
			-DBUILD_PYTHON_BINDINGS=OFF -DRUN_DOXYGEN=OFF \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_INSTALL_PREFIX=$(CURDIR) -DCMAKE_INSTALL_BINDIR=. && \
		$(MAKE) -j$$(nproc) handlegraph_to_csr'

csr_to_adj:
	$(CXX) $(CXXFLAGS) src/csr_to_adj.cpp -o $@

build_pch:
	$(CXX) $(CXXFLAGS) \
		-I ext/Parallel-Contraction-Hierarchy/ \
		-I ext/Parallel-Contraction-Hierarchy/parlaylib/include/ \
		src/CHoverrides/build_pch.cpp -o $@

csr_to_char_graphs_matchN:
	@mkdir -p char_graph_build_matchN
	@cd char_graph_build_matchN && CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" \
		cmake .. \
		-DBUILD_HANDLEGRAPH_TO_CSR=OFF -DBUILD_CSR_TO_CHAR_GRAPHS=ON -DBUILD_PROTOBUF_TO_CSR=OFF \
		-DDAG=0 -DALPHABET_SIZE=4 \
		-DBUILD_PYTHON_BINDINGS=OFF -DRUN_DOXYGEN=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$(CURDIR) -DCMAKE_INSTALL_BINDIR=. && \
		$(MAKE) -j$$(nproc) csr_to_char_graphs
	@mv $(CURDIR)/csr_to_char_graphs $(CURDIR)/csr_to_char_graphs_matchN

csr_to_char_graphs_mismatchN:
	@mkdir -p char_graph_build_mismatchN
	@cd char_graph_build_mismatchN && CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" \
		cmake .. \
		-DBUILD_HANDLEGRAPH_TO_CSR=OFF -DBUILD_CSR_TO_CHAR_GRAPHS=ON -DBUILD_PROTOBUF_TO_CSR=OFF \
		-DDAG=0 -DALPHABET_SIZE=5 \
		-DBUILD_PYTHON_BINDINGS=OFF -DRUN_DOXYGEN=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=$(CURDIR) -DCMAKE_INSTALL_BINDIR=. && \
		$(MAKE) -j$$(nproc) csr_to_char_graphs
	@mv $(CURDIR)/csr_to_char_graphs $(CURDIR)/csr_to_char_graphs_mismatchN

# Requires a conda environment providing protobuf development files.
protobuf_to_csr:
	@bash -c 'mkdir -p protobuf_to_csr_build && cd protobuf_to_csr_build && \
		CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" \
		cmake .. \
			-DPROTOBUF_DIR="$$CONDA_PREFIX" \
			-DBUILD_HANDLEGRAPH_TO_CSR=OFF \
			-DBUILD_CSR_TO_CHAR_GRAPHS=OFF \
			-DBUILD_PROTOBUF_TO_CSR=ON \
			-DBUILD_PYTHON_BINDINGS=OFF -DRUN_DOXYGEN=OFF \
			-DCMAKE_BUILD_TYPE=Release \
			-DCMAKE_INSTALL_PREFIX=$(CURDIR) -DCMAKE_INSTALL_BINDIR=. && \
		$(MAKE) -j$$(nproc) protobuf_to_csr'

# ---------------------------------------------------------------------------
# Score-only aligners (general graphs)
# ---------------------------------------------------------------------------

serialParSGA_matchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=0 $(PAR_INCLUDES) src/serialParSGA.cpp -fopenmp -o $@
serialParSGA_mismatchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=0 $(PAR_INCLUDES) src/serialParSGA.cpp -fopenmp -o $@
ParSGA_matchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=0 $(PAR_INCLUDES) src/ParSGA.cpp -fopenmp -o $@
ParSGA_mismatchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=0 $(PAR_INCLUDES) src/ParSGA.cpp -fopenmp -o $@
ConSGA_matchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=0 $(CON_INCLUDES) src/ConSGA.cpp -o $@ -fopenmp
ConSGA_mismatchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=0 $(CON_INCLUDES) src/ConSGA.cpp -o $@ -fopenmp
all_gen_graph: serialParSGA_matchN serialParSGA_mismatchN ParSGA_matchN ParSGA_mismatchN ConSGA_matchN ConSGA_mismatchN

ConSGA_mismatchNpapi:
	$(CXX) $(CXXFLAGS) -Wall -Wextra -DALPHABET_SIZE=5 -DDAG=0 $(CON_INCLUDES) \
		src/ConSGA.cpp -o $@ -fopenmp $(PAPI_FLAGS)

# ---------------------------------------------------------------------------
# Score-only aligners (DAGs)
# ---------------------------------------------------------------------------

serialParSGA_matchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=1 $(PAR_INCLUDES) src/serialParSGA.cpp -fopenmp -o $@
serialParSGA_mismatchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=1 $(PAR_INCLUDES) src/serialParSGA.cpp -fopenmp -o $@
ParSGA_matchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=1 $(PAR_INCLUDES) src/ParSGA.cpp -fopenmp -o $@
ParSGA_mismatchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=1 $(PAR_INCLUDES) src/ParSGA.cpp -fopenmp -o $@
ConSGA_matchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=1 $(CON_INCLUDES) src/ConSGA.cpp -o $@ -fopenmp
ConSGA_mismatchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=1 $(CON_INCLUDES) src/ConSGA.cpp -o $@ -fopenmp
all_dag: serialParSGA_matchN_dag serialParSGA_mismatchN_dag ParSGA_matchN_dag ParSGA_mismatchN_dag \
	ConSGA_matchN_dag ConSGA_mismatchN_dag PaSGAL PaSGAL_match

# ---------------------------------------------------------------------------
# Traceback-inclusive aligners
#   Default mode verifies path scores against the score-only reference.
#   Pass -b for paper-style per-read timing (skips the first read in the mean).
# ---------------------------------------------------------------------------

ConSGA_traceback_matchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=0 $(CON_INCLUDES) src/ConSGA_traceback.cpp -o $@ -fopenmp
ConSGA_traceback_mismatchN:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=0 $(CON_INCLUDES) src/ConSGA_traceback.cpp -o $@ -fopenmp
ConSGA_traceback_matchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=1 $(CON_INCLUDES) src/ConSGA_traceback.cpp -o $@ -fopenmp
ConSGA_traceback_mismatchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=1 $(CON_INCLUDES) src/ConSGA_traceback.cpp -o $@ -fopenmp
ParSGA_traceback_matchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=4 -DDAG=1 $(PAR_INCLUDES) src/ParSGA_traceback.cpp -fopenmp -o $@
ParSGA_traceback_mismatchN_dag:
	$(CXX) $(CXXFLAGS) -DALPHABET_SIZE=5 -DDAG=1 $(PAR_INCLUDES) src/ParSGA_traceback.cpp -fopenmp -o $@

# Same as above with an 800 GiB fast-path store (needed for Opossum-scale graphs).
ConSGA_traceback_mismatchN_dag_800gb:
	$(CXX) $(CXXFLAGS) $(FAST_STORE_800GB) -DALPHABET_SIZE=5 -DDAG=1 $(CON_INCLUDES) \
		src/ConSGA_traceback.cpp -o $@ -fopenmp
ParSGA_traceback_mismatchN_dag_800gb:
	$(CXX) $(CXXFLAGS) $(FAST_STORE_800GB) -DALPHABET_SIZE=5 -DDAG=1 $(PAR_INCLUDES) \
		src/ParSGA_traceback.cpp -fopenmp -o $@

all_traceback_dag: ConSGA_traceback_matchN_dag ConSGA_traceback_mismatchN_dag \
	ParSGA_traceback_matchN_dag ParSGA_traceback_mismatchN_dag

# ---------------------------------------------------------------------------
# PaSGAL (external; optional cost-model override for match-among-N)
# ---------------------------------------------------------------------------

PaSGAL:
	@bash -c 'rm -rf ext/PaSGAL/build && mkdir -p ext/PaSGAL/build && cd ext/PaSGAL/build && \
		CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" \
		cmake .. \
			-DSIMD_SUPPORT=none \
			-DPROTOBUF_DIR="$$CONDA_PREFIX" \
			-DBUILD_HANDLEGRAPH_TO_CSR=OFF \
			-DBUILD_CSR_TO_CHAR_GRAPHS=OFF \
			-DBUILD_PROTOBUF_TO_CSR=ON \
			-DBUILD_PYTHON_BINDINGS=OFF -DRUN_DOXYGEN=OFF \
			-DCMAKE_BUILD_TYPE=Release && \
		$(MAKE) -j$$(nproc) && cp PaSGAL $(CURDIR)/'

PaSGAL_match:
	@bash -c 'cp ext/PaSGAL/src/include/align.hpp ext/PaSGAL/src/include/align.hpp.bak && \
		cp src/PaSGALoverrides/align.hpp ext/PaSGAL/src/include/align.hpp && \
		rm -rf ext/PaSGAL/build_match && mkdir -p ext/PaSGAL/build_match && \
		cd ext/PaSGAL/build_match && \
		CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" \
		cmake .. \
			-DSIMD_SUPPORT=none \
			-DPROTOBUF_DIR="$$CONDA_PREFIX" \
			-DBUILD_HANDLEGRAPH_TO_CSR=OFF \
			-DBUILD_CSR_TO_CHAR_GRAPHS=OFF \
			-DBUILD_PROTOBUF_TO_CSR=ON \
			-DBUILD_PYTHON_BINDINGS=OFF -DRUN_DOXYGEN=OFF \
			-DCMAKE_BUILD_TYPE=Release && \
		$(MAKE) -j$$(nproc) && cp PaSGAL $(CURDIR)/PaSGAL_match && \
		mv $(CURDIR)/ext/PaSGAL/src/include/align.hpp.bak \
		   $(CURDIR)/ext/PaSGAL/src/include/align.hpp'

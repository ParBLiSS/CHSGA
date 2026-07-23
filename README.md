# CHSGA

CHSGA (also referred to as ConSGA) is a parallel sequence-to-graph aligner based on contraction hierarchies. This repository contains the source used for the CHSGA paper experiments, including:

- **CHSGA / ConSGA** score-only and traceback-inclusive alignment
- **ParSGA** score-only and traceback-inclusive alignment (traceback drivers live here; they are not in [ParBLiSS/ParSGA](https://github.com/ParBLiSS/ParSGA))
- Preprocessing utilities (handlegraph → CSR, CSR → character graphs, contraction hierarchy construction)
- Optional [PaSGAL](https://github.com/ParBLiSS/PaSGAL) builds for serial baseline comparison

Graph and read datasets are not included (they are large). Provide your own inputs in the formats described below.

## Requirements

- C++17-capable `g++` with OpenMP
- CMake ≥ 3.10
- `zlib` development headers
- Git (for submodules)
- For PaSGAL / protobuf conversion: a Conda environment with protobuf development files (`CONDA_PREFIX` must be set)
- For `handlegraph_to_csr`: [libbdsg](https://github.com/vgteam/libbdsg) dependencies via the `ext/libbdsg` submodule

Large-graph traceback (hundreds of GB of score/backpointer storage) needs a machine with sufficient RAM (on the order of 1–3 TB for the largest paper graphs).

## Clone

```bash
git clone --recursive git@github.com:ParBLiSS/CHSGA.git
cd CHSGA
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build

Score-only DAG aligners (alphabet size 5, N treated as a mismatch against A/C/G/T):

```bash
make ConSGA_mismatchN_dag ParSGA_mismatchN_dag -j
```

Traceback-inclusive DAG aligners:

```bash
make all_traceback_dag -j
```

For graphs whose in-memory score/backpointer store exceeds the default 400 GiB fast-path cap (e.g. Opossum-scale inputs), build the 800 GiB variants:

```bash
make ConSGA_traceback_mismatchN_dag_800gb ParSGA_traceback_mismatchN_dag_800gb -j
```

Other useful targets:

| Target | Description |
|--------|-------------|
| `ConSGA_matchN_dag` / `ParSGA_matchN_dag` | Score-only; N matches N |
| `build_pch` | Build contraction hierarchy from a string-labeled adjacency graph |
| `csr_to_char_graphs_mismatchN` | Expand CSR string graphs to ParSGA character-graph inputs |
| `PaSGAL` / `PaSGAL_match` | Serial PaSGAL baseline (optional cost-model override) |

## Input formats

### CHSGA / ConSGA

```text
-I <char.adj>          character-labeled adjacency graph
-i <string.adj>        string-labeled adjacency graph
-c <string.out>        contracted hierarchy (from build_pch)
-r <reads.fastq>       reads (FASTQ or one sequence per line)
-l <char.labels>       concatenated vertex labels
-o <string.label_offsets>  per-vertex label offsets
```

Optional traceback flags:

```text
-b                     benchmark mode (paper-style timing; skips first read in the mean)
-k <rows>              checkpoint interval (default 500)
-d <dir>               checkpoint directory
```

Example:

```bash
export PARLAY_NUM_THREADS=128
./ConSGA_traceback_mismatchN_dag -b -k 500 -d /tmp/consga_ckpts \
  -I consga_input/graphchar.adj \
  -i consga_input/graphstring.adj \
  -c consga_input/graphstring.out \
  -r reads.fastq \
  -l consga_input/graphchar.labels \
  -o consga_input/graphstring.label_offsets
```

Without `-b`, the driver verifies that each reconstructed path score matches the score-only reference and prints CIGAR strings.

### ParSGA

Positional arguments match ParSGA:

```text
[-b] [-k rows] [-d ckpt_dir] <csr_prefix> <unused_output> <reads.fastq> <num_threads>
```

Example:

```bash
export OMP_NUM_THREADS=128
./ParSGA_traceback_mismatchN_dag -b -k 500 -d /tmp/parsga_ckpts \
  parsga_input/graphchar /dev/null reads.fastq 128
```

### Preprocessing sketch

1. Convert a handlegraph / VG graph to CSR (`handlegraph_to_csr` or `protobuf_to_csr`).
2. Build string- and character-labeled adjacency inputs (`csr_to_adj`, `csr_to_char_graphs_*`).
3. Build the contraction hierarchy: `./build_pch <string.adj> <string.out>`.
4. Run CHSGA / ParSGA / PaSGAL as above.

Exact file naming for a given dataset should match the prefixes expected by each tool.

## Traceback implementation notes

- **CHSGA** fast path keeps the full DP score matrix for a read when it fits under the configured byte cap, then reconstructs an optimal path by score equality (no separate per-cell backpointer matrix on that path).
- Above the cap, both CHSGA and ParSGA use NVMe-backed checkpointed traceback with segment recompute.
- Parallel first-touch of very large score matrices (≥ 300 GiB) is enabled automatically in CHSGA to reduce page-fault serialization on large graphs.

## Layout

```text
src/                   CHSGA, ParSGA wrappers, traceback, preprocessing
src/ParSGAoverrides/   headers shared with / overriding ParSGA
src/CHoverrides/       contraction-hierarchy build driver
src/PaSGALoverrides/   optional PaSGAL cost-model patch
ext/                   git submodules (PCH, ParSGA, PaSGAL, libbdsg)
```

## License

See individual submodule repositories for their licenses. Code in `src/` is provided for reproducing the associated publication; contact the authors for licensing questions if redistributing.

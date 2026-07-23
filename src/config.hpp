#pragma once

#ifndef ALPHABET_SIZE
#error "Define ALPHABET_SIZE via -DALPHABET_SIZE=4 or 5"
#endif

#if ALPHABET_SIZE == 5
static const int ALPHABET_SIZE_CONST = 5;
static const char bases[ALPHABET_SIZE_CONST] = {'T','C','G','A','N'};
#elif ALPHABET_SIZE == 4
static const int ALPHABET_SIZE_CONST = 4;
static const char bases[ALPHABET_SIZE_CONST] = {'T','C','G','A'};
#else
#error "ALPHABET_SIZE must be 4 or 5"
#endif

inline int8_t base_index(char base) {
  switch (base) {
    case 'T': return 0;
    case 'C': return 1;
    case 'G': return 2;
    case 'A': return 3;
    case 'N': return 4;
    default: return -1;
  }
}

#pragma once

#ifndef DAG
#error "Define DAG via -DDAG=0 or -DDAG=1"
#endif

#if DAG == 1
static constexpr bool dag = true;
#elif DAG == 0
static constexpr bool dag = false;
#else
#error "DAG must be 0 or 1"
#endif
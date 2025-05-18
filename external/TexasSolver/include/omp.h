#ifndef OMP_H
#define OMP_H

// This is a mock OpenMP header for platforms where OpenMP is not available

#ifdef DISABLE_OMP

// OpenMP function definitions
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_get_max_threads() 1
#define omp_get_num_procs() 1
#define omp_in_parallel() 0

// OpenMP Runtime Library functions
static inline void omp_set_num_threads(int num_threads) { (void)num_threads; }
static inline void omp_set_dynamic(int dynamic) { (void)dynamic; }
static inline void omp_set_nested(int nested) { (void)nested; }

// OpenMP Locks
typedef void* omp_lock_t;
static inline void omp_init_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_set_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_unset_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_destroy_lock(omp_lock_t *lock) { (void)lock; }
static inline int omp_test_lock(omp_lock_t *lock) { (void)lock; return 1; }

// OpenMP Timing
static inline double omp_get_wtime(void) { return 0.0; }
static inline double omp_get_wtick(void) { return 0.0; }

#endif // DISABLE_OMP

#endif // OMP_H 
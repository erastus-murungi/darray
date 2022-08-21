#include "strassen.h"
#include "utils.h"

#include <assert.h>
#include <errno.h>
#include <immintrin.h>
#include <locale.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/types.h>
#include <wchar.h>

#ifdef PARALLEL
#include <omp.h>
#endif

#define BUFLEN 256
#define MAX_NCHARS 32
char buf[BUFLEN];

static const char ERR_MSG_DIMS_UNEQUAL[] =
    "adding requires matrix dimensions to be equal";
static const char ERR_MSG_POINTER_TO_DARRAY_IS_NULL[] =
    "pointer to darray is null";

static void exit_if_null(const void *ptr, const int line_num, const char *msg) {
  if (ptr == NULL) {
    REPORT_ERROR_AND_EXIT((ptr == NULL), line_num, __FILE__, "%s", msg);
  }
}

static void exit_if_violates_bounds(const ssize_t val, const ssize_t bound,
                                    const int line_num, const char *msg) {
  if (val >= bound) {
    REPORT_ERROR_AND_EXIT((val >= bound), line_num, __FILE__,
                          "%ld must be < %ld: %s", val, bound, msg);
  }
}

static void exit_if_no_mem_error(int memalign_ret, const char *msg,
                                 const int line_num) {
  if (memalign_ret != 0) {
    strerror_r(errno, buf, BUFLEN);
    REPORT_ERROR_AND_EXIT((ptr == NULL), line_num, __FILE__, "%s: %s", buf,
                          msg);
  }
}

static void exit_if_no_mem_error_malloc(void *ptr, const char *msg,
                                        const int line_num) {
  if (ptr == NULL) {
    strerror_r(errno, buf, BUFLEN);
    REPORT_ERROR_AND_EXIT((ptr == NULL), line_num, __FILE__, "%s: %s", buf,
                          msg);
  }
}

static void exit_if_empty_array(const darray *da, const char *msg) {
  if (darray_get_size(da) == 0) {
    REPORT_ERROR_AND_EXIT((darray_get_size(da) == 0), __LINE__, __FILE__,
                          "len(darray) = 0: %s", msg);
  }
}

void exit_if_array_access_checks_fail(const darray *da, ssize_t row,
                                      ssize_t col) {
  exit_if_null(da, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  exit_if_null(da->elems, __LINE__, "null pointer to underlying doubles array");
  exit_if_violates_bounds(row, da->dims.n_rows, __LINE__,
                          "row index out of bounds");
  exit_if_violates_bounds(col, da->dims.n_cols, __LINE__,
                          "column index out of bounds");
}

dims exit_if_matrix_multiply_dims_mismatch(const dims dims_a,
                                           const dims dims_b) {
  if (dims_a.n_cols != dims_b.n_rows) {
    REPORT_ERROR_AND_EXIT(
        dims_a.n_cols != dims_b.n_rows, __LINE__, __FILE__,
        "matrix multiply dimensions mismatch: got (%ld x %ld) * (%ld x %ld)",
        dims_a.n_rows, dims_a.n_cols, dims_b.n_rows, dims_b.n_cols);
  }
  dims dims_c = (dims){.n_rows = dims_a.n_rows, .n_cols = dims_b.n_cols};
  return dims_c;
}

void exit_if_dims_not_equal(const dims dims_a, const dims dims_b,
                            const char *msg) {
  if (dims_a.n_rows != dims_b.n_rows || dims_a.n_cols != dims_b.n_cols) {
    REPORT_ERROR_AND_EXIT(
        dims_a.n_rows != dims_b.n_rows || dims_a.n_cols != dims_b.n_cols,
        __LINE__, __FILE__, "(%ld, %ld) != (%ld, %ld) : %s", dims_a.n_rows,
        dims_a.n_cols, dims_b.n_rows, dims_b.n_cols, msg);
  }
}

void exit_if_new_dims_smaller(dims dims_old, dims dims_new, const char *msg,
                              const int line_num) {
  if (dims_old.n_cols > dims_new.n_cols) {
    REPORT_ERROR_AND_EXIT(dims_old.n_cols > dims_new.n_cols, line_num, __FILE__,
                          "%s: %s", "dims_old.n_cols > dims_new.n_cols", msg);
  }
  if (dims_old.n_rows > dims_new.n_rows) {
    REPORT_ERROR_AND_EXIT(dims_old.n_rows > dims_new.n_rows, line_num, __FILE__,
                          "%s: %s", "dims_old.n_rows > dims_new.n_rows", msg);
  }
}

void exit_if_new_dims_greater(dims dims_old, dims dims_new, const char *msg,
                              const int line_num) {
  if (dims_old.n_cols < dims_new.n_cols) {
    REPORT_ERROR_AND_EXIT(dims_old.n_cols < dims_new.n_cols, line_num, __FILE__,
                          "%s: %s", "dims_old.n_cols < dims_new.n_cols", msg);
  }
  if (dims_old.n_rows < dims_new.n_rows) {
    REPORT_ERROR_AND_EXIT(dims_old.n_rows < dims_new.n_rows, line_num, __FILE__,
                          "%s: %s", "dims_old.n_rows < dims_new.n_rows", msg);
  }
}

ssize_t darray_get_size(const darray *da) {
  exit_if_null(da, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  return da->dims.n_cols * da->dims.n_rows;
}

bool darray_is_square(const darray *da) {
  return da->dims.n_rows == da->dims.n_cols;
}

static inline double darray_view_get_elem_no_checks(const darray_view *dv,
                                                    ssize_t row_num,
                                                    ssize_t col_num) {
  return dv->d->elems[((row_num + dv->row_start) * dv->d->dims.n_cols) +
                      col_num + dv->col_start];
}
static inline double *darray_view_get_elem_ptr_no_checks(const darray_view *dv,
                                                         const ssize_t row_num,
                                                         ssize_t col_num) {
  return &dv->d->elems[((row_num + dv->row_start) * dv->d->dims.n_cols) +
                       col_num + dv->col_start];
}

static inline double darray_view_set_elem_no_checks(const darray_view *dv,
                                                    ssize_t row_num,
                                                    ssize_t col_num,
                                                    double elem) {
  return dv->d->elems[((row_num + dv->row_start) * dv->d->dims.n_cols) +
                      col_num + dv->col_start] = elem;
}

darray *darray_new_uninitialized(const ssize_t n_rows, const ssize_t n_cols) {
  const ssize_t n_elems = n_cols * n_rows;
  exit_if_violates_bounds(n_elems, MAX_ELEMS, __LINE__, "too many elements");
  darray *da = malloc(sizeof(darray));
  exit_if_no_mem_error_malloc(da, "could not allocate memory for darray",
                              __LINE__);
  double *a = malloc(sizeof(double) * n_elems);
  exit_if_no_mem_error_malloc(
      a, "cannot not allocate memory for underlying doubles array", __LINE__);
  *da = (darray){.elems = a, .dims = {.n_rows = n_rows, .n_cols = n_cols}};
  return da;
}

darray *darray_new_zeroed(ssize_t n_rows, ssize_t n_cols) {
  darray *d = darray_new_uninitialized(n_rows, n_cols);
  memset(d->elems, '\0', darray_get_size(d) * sizeof(double));
  return d;
}

darray *darray_new_initialized_from_doubles(const ssize_t n_rows,
                                            const ssize_t n_cols, double **ds) {
  darray *da = darray_new_uninitialized(n_rows, n_cols);
  memcpy(da->elems, ds, sizeof(double) * darray_get_size(da));
  return da;
}

void darray_set(darray *d, double to) {
  exit_if_null(d, __LINE__, "darray is null");
  exit_if_null(d->elems, __LINE__, "darray elems pointer is null");

  const ssize_t len = darray_get_size(d);

  double *a = d->elems;
#pragma clang loop vectorize(enable) interleave(enable)
  for (ssize_t i = 0; i < len; i++) {
    a[i] = to;
  }
}

darray_view *darray_view_new(darray *da, ssize_t row_start, ssize_t row_end,
                             ssize_t col_start, ssize_t col_end) {
  darray_view *dv = malloc(sizeof(darray_view));
  exit_if_no_mem_error_malloc(dv, "could not create darray_view struct",
                              __LINE__);
  *dv = (darray_view){.d = da,
                      .row_start = row_start,
                      .row_end = row_end,
                      .col_start = col_start,
                      .col_end = col_end};
  return dv;
}

void darray_free(darray *da) {
  exit_if_null(da, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  exit_if_null(da->elems, __LINE__,
               "cannot free null pointer to underlying doubles array");
  free(da->elems);
  free(da);
}

void darray_view_free(darray_view *dv) {
  exit_if_null(dv, __LINE__, "pointer to darray_view is null");
  free(dv);
}

inline double darray_get_elem_no_checks(const darray *da, ssize_t row,
                                        ssize_t col) {
  const ssize_t where = (row * da->dims.n_cols + col);
  return da->elems[where];
}

static inline double *darray_get_elem_ptr_no_checks(const darray *da,
                                                    ssize_t row, ssize_t col) {
  const ssize_t where = (row * da->dims.n_cols + col);
  return da->elems + where;
}

bool darray_equal(darray *da_this, darray *da_that) {
  if (da_that == NULL || da_this == NULL) {
    return false;
  }

  if (da_that->dims.n_cols != da_this->dims.n_cols ||
      da_that->dims.n_rows != da_this->dims.n_rows) {
    printf("here\n");
    return false;
  }

#pragma clang loop vectorize(enable) interleave(enable)
  for (ssize_t i = 0; i < da_this->dims.n_rows; i++) {
    for (ssize_t j = 0; j < da_this->dims.n_cols; j++) {
      if (!approximately_equal(darray_get_elem_no_checks(da_this, i, j),
                               darray_get_elem_no_checks(da_that, i, j))) {
        return false;
      }
    }
  }
  return true;
}

static inline void darray_set_elem_no_checks(darray *da, ssize_t row,
                                             ssize_t col, double value) {
  const ssize_t where = (row * da->dims.n_cols + col);
  da->elems[where] = value;
}

double darray_get_item(darray *da, ssize_t row, ssize_t col) {
  exit_if_array_access_checks_fail(da, row, col);
  return darray_get_elem_no_checks(da, row, col);
}

void darray_set_item(darray *da, ssize_t row, ssize_t col, double value) {
  exit_if_array_access_checks_fail(da, row, col);
  darray_set_elem_no_checks(da, row, col, value);
}

bool darray_is_empty(darray *da) { return darray_get_size(da) == 0; }

darray *darray_copy(darray *da) {
  darray *da_copy = darray_new_uninitialized(da->dims.n_rows, da->dims.n_cols);
  memcpy(da_copy->elems, da->elems,
         sizeof(double) * da->dims.n_cols * da->dims.n_rows);
  da_copy->dims.n_cols = da->dims.n_cols;
  da_copy->dims.n_rows = da->dims.n_rows;
  return da_copy;
}

darray *darray_copy_into(darray *da, darray *da_copy) {
  exit_if_null(da, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  exit_if_null(da_copy, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  exit_if_new_dims_smaller(da->dims, da_copy->dims,
                           "can't copy into smaller darray", __LINE__);
  double *s = da_copy->elems;
  double *t = da->elems;
  for (ssize_t i = 0; i < da->dims.n_rows;
       i++, s += (da_copy->dims.n_cols), t += (da->dims.n_cols)) {
    memcpy(s, t, sizeof(double) * da->dims.n_cols);
  }
  return da_copy;
}

double darray_find_max(const darray *da) {
  exit_if_null(da, __LINE__, "cannot find max of null darray");
  exit_if_empty_array(da, "cannot find max of array with len 0");
  const ssize_t size = darray_get_size(da);
  double max_val = da->elems[0];
#ifdef PARALLEL
#pragma omp parallel for reduction(max : max_val)
#endif
  for (ssize_t i = 0; i < size; i++) {
    max_val = MAX(max_val, da->elems[i]);
  }
  return max_val;
}

double darray_find_min(const darray *da) {
  exit_if_null(da, __LINE__, "cannot find max of null darray");
  exit_if_empty_array(da, "cannot find min of array with len 0");
  const ssize_t size = darray_get_size(da);
  double min_val = da->elems[0];
#ifdef PARALLEL
#pragma omp parallel for reduction(min : min_val)
#endif
  for (ssize_t i = 0; i < size; i++) {
    min_val = MIN(min_val, da->elems[i]);
  }
  return min_val;
}

darray *darray_add(const darray *da, const darray *db) {
  exit_if_null(da, __LINE__, "first matrix is null");
  exit_if_null(db, __LINE__, "second matrix is null");
  exit_if_dims_not_equal(da->dims, db->dims, ERR_MSG_DIMS_UNEQUAL);
  darray *dc = darray_new_uninitialized(da->dims.n_rows, da->dims.n_cols);

  ssize_t len = darray_get_size(da);

  double *a = da->elems;
  double *b = db->elems;
  double *c = dc->elems;

  ssize_t i = 0;
  for (; i < (len & ~0x3); i += 4) {
    const __m256d ma = _mm256_load_pd(&a[i]);
    const __m256d mb = _mm256_load_pd(&b[i]);
    const __m256d mc = _mm256_add_pd(ma, mb);
    _mm256_stream_pd(&c[i], mc);
  }
  for (; i < len; i++) {
    c[i] = a[i] + b[i];
  }
  return dc;
}

darray *darray_view_add(const darray_view *dva, const darray_view *dvb) {
  exit_if_null(dva, __LINE__, "first matrix is null");
  exit_if_null(dvb, __LINE__, "second matrix is null");
  const dims dims_a = (dims){.n_rows = (dva->row_end - dva->row_start),
                             .n_cols = (dva->col_end - dva->col_start)};
  const dims dims_b = (dims){.n_rows = (dvb->row_end - dvb->row_start),
                             .n_cols = (dvb->col_end - dvb->col_start)};
  exit_if_dims_not_equal(dims_a, dims_b, ERR_MSG_DIMS_UNEQUAL);
  darray *dc = darray_new_uninitialized(dims_a.n_rows, dims_b.n_cols);

  const ssize_t n_cols = dims_a.n_cols;
  const ssize_t n_rows = dims_a.n_rows;

  for (ssize_t row = 0; row < n_rows; row++) {
    double *a = darray_view_get_elem_ptr_no_checks(dva, row, 0);
    double *b = darray_view_get_elem_ptr_no_checks(dvb, row, 0);
    double *c = dc->elems + (row * n_cols);

    ssize_t i = 0;
    for (; i < (n_cols & ~0x3); i += 4) {
      const __m256d ma = _mm256_loadu_pd(&a[i]);
      const __m256d mb = _mm256_loadu_pd(&b[i]);
      const __m256d mc = _mm256_add_pd(ma, mb);
      _mm256_stream_pd(&c[i], mc);
    }
    for (; i < n_cols; i++) {
      c[i] = a[i] + b[i];
    }
  }
  return dc;
}

void exit_if_darray_view_subadd_checks_fail(const darray_view *dva,
                                            const darray_view *dvb,
                                            const darray_view *dvc) {
  exit_if_null(dva, __LINE__, "first matrix is null");
  exit_if_null(dvb, __LINE__, "second matrix is null");
  const dims dims_a = (dims){.n_rows = (dva->row_end - dva->row_start),
                             .n_cols = (dva->col_end - dva->col_start)};
  const dims dims_b = (dims){.n_rows = (dvb->row_end - dvb->row_start),
                             .n_cols = (dvb->col_end - dvb->col_start)};
  const dims dims_c = (dims){.n_rows = (dvc->row_end - dvc->row_start),
                             .n_cols = (dvc->col_end - dvc->col_start)};
  exit_if_dims_not_equal(dims_a, dims_b, ERR_MSG_DIMS_UNEQUAL);
  exit_if_dims_not_equal(dims_a, dims_c, ERR_MSG_DIMS_UNEQUAL);
}

#define GRAINSIZE_ADD_OUTER_LOOP (1000)
#define GRAINSIZE_ADD_INNER_LOOP (1000)
void darray_view_add_into(const darray_view *dva, const darray_view *dvb,
                          const darray_view *dvc) {
  exit_if_darray_view_subadd_checks_fail(dva, dvb, dvc);
  const ssize_t n_cols = dva->col_end - dva->col_start;
  const ssize_t n_rows = dva->row_end - dva->row_start;

#pragma omp parallel for grainsize(GRAINSIZE_ADD_OUTER_LOOP)
  for (ssize_t row = 0; row < n_rows; row++) {
    double *a = darray_view_get_elem_ptr_no_checks(dva, row, 0);
    double *b = darray_view_get_elem_ptr_no_checks(dvb, row, 0);
    double *c = darray_view_get_elem_ptr_no_checks(dvc, row, 0);

    ssize_t i = 0;
#pragma omp parallel for grainsize(GRAINSIZE_ADD_INNER_LOOP)
    for (; i < (n_cols & ~0x3); i += 4) {
      const __m256d ma = _mm256_loadu_pd(&a[i]);
      const __m256d mb = _mm256_loadu_pd(&b[i]);
      const __m256d mc = _mm256_add_pd(ma, mb);
      _mm256_storeu_pd(&c[i], mc);
    }
    for (; i < n_cols; i++) {
      c[i] = a[i] + b[i];
    }
  }
}

darray *darray_view_sub(const darray_view *dva, const darray_view *dvb) {
  exit_if_null(dva, __LINE__, "first matrix is null");
  exit_if_null(dvb, __LINE__, "second matrix is null");
  const dims dims_a = (dims){.n_rows = (dva->row_end - dva->row_start),
                             .n_cols = (dva->col_end - dva->col_start)};
  const dims dims_b = (dims){.n_rows = (dvb->row_end - dvb->row_start),
                             .n_cols = (dvb->col_end - dvb->col_start)};
  exit_if_dims_not_equal(dims_a, dims_b, ERR_MSG_DIMS_UNEQUAL);
  darray *dc = darray_new_uninitialized(dims_a.n_rows, dims_b.n_cols);

  const ssize_t n_cols = dims_a.n_cols;
  const ssize_t n_rows = dims_a.n_rows;

  for (ssize_t row = 0; row < n_rows; row++) {
    double *a = darray_view_get_elem_ptr_no_checks(dva, row, 0);
    double *b = darray_view_get_elem_ptr_no_checks(dvb, row, 0);
    double *c = dc->elems + (row * n_cols);

    ssize_t i = 0;
    for (; i < (n_cols & ~0x3); i += 4) {
      const __m256d ma = _mm256_load_pd(&a[i]);
      const __m256d mb = _mm256_load_pd(&b[i]);
      const __m256d mc = _mm256_sub_pd(ma, mb);
      _mm256_stream_pd(&c[i], mc);
    }
    for (; i < n_cols; i++) {
      c[i] = a[i] + b[i];
    }
  }
  return dc;
}

void darray_view_sub_into(const darray_view *dva, const darray_view *dvb,
                          const darray_view *dvc) {
  exit_if_darray_view_subadd_checks_fail(dva, dvb, dvc);
  const ssize_t n_cols = dva->d->dims.n_cols;
  const ssize_t n_rows = dva->d->dims.n_rows;

  for (ssize_t row = 0; row < n_rows; row++) {
    double *a = darray_view_get_elem_ptr_no_checks(dva, row, 0);
    double *b = darray_view_get_elem_ptr_no_checks(dvb, row, 0);
    double *c = darray_view_get_elem_ptr_no_checks(dvc, row, 0);

    ssize_t i = 0;
    for (; i < (n_cols & ~0x3); i += 4) {
      const __m256d ma = _mm256_loadu_pd(&a[i]);
      const __m256d mb = _mm256_loadu_pd(&b[i]);
      const __m256d mc = _mm256_sub_pd(ma, mb);
      _mm256_stream_pd(&c[i], mc);
    }
    for (; i < n_cols; i++) {
      c[i] = a[i] + b[i];
    }
  }
}

dims darray_view_get_dims(const darray_view *dv) {
  const ssize_t n_rows = (dv->row_end - dv->row_start);
  const ssize_t n_cols = (dv->col_end - dv->col_start);
  const dims dims_ret = {.n_rows = n_rows, .n_cols = n_cols};
  return dims_ret;
}

void darray_view_fmadd(const darray_view *x11, const darray_view *y11,
                       const darray_view *x12, const darray_view *y21,
                       darray_view *z11) {
  exit_if_null(x11, __LINE__, "x11 matrix is null");
  exit_if_null(y11, __LINE__, "y11 matrix is null");
  exit_if_null(x12, __LINE__, "x12 matrix is null");
  exit_if_null(y21, __LINE__, "y21 matrix is null");

  exit_if_dims_not_equal(darray_view_get_dims(x11), darray_view_get_dims(x12),
                         "x11 and x12 dims must be equal");
  exit_if_dims_not_equal(darray_view_get_dims(y11), darray_view_get_dims(y21),
                         "y11 and y21 dims must be equal");
  dims dims_z11_expected = exit_if_matrix_multiply_dims_mismatch(
      darray_view_get_dims(x11), darray_view_get_dims(y11));
  exit_if_dims_not_equal(dims_z11_expected, darray_view_get_dims(z11), "exp");

  const ssize_t n_rows_view_A = x11->row_end - x11->row_start;
  const ssize_t n_cols_view_A = x11->col_end - x11->col_start;
  const ssize_t n_cols_view_B = y11->col_end - y11->col_start;

  ssize_t i, j, k;
  // for each element in a row of A
  for (i = 0; i < n_rows_view_A; i++) {
    // for each element of a row of A
    for (j = 0; j < n_cols_view_A; j++) {
      double temp = .0;
      for (k = 0; k < n_cols_view_B; k++) {
        // for each element in a column of B
        const double x11_ik = darray_view_get_elem_no_checks(x11, i, k);
        const double y11_kj = darray_view_get_elem_no_checks(y11, k, j);
        const double x12_ik = darray_view_get_elem_no_checks(x12, i, k);
        const double y21_kj = darray_view_get_elem_no_checks(y21, k, j);
        temp += (x11_ik * y11_kj) + (x12_ik + y21_kj);
      }
      const double C_ij = darray_view_get_elem_no_checks(z11, i, j);
      darray_view_set_elem_no_checks(z11, i, j, temp + C_ij);
    }
  }
}

darray *darray_sub(darray *da, darray *db) {
  exit_if_null(da, __LINE__, "first matrix is null");
  exit_if_null(db, __LINE__, "second matrix is null");
  exit_if_dims_not_equal(da->dims, db->dims,
                         "adding requires matrix dimensions to be equal");
  darray *dc = darray_new_uninitialized(da->dims.n_rows, da->dims.n_cols);

  const ssize_t len = darray_get_size(da);

  double *a = da->elems;
  double *b = db->elems;
  double *c = dc->elems;
  ssize_t i = 0;
  for (; i < (len & ~0x3); i += 4) {
    const __m256d ma = _mm256_loadu_pd(&a[i]);
    const __m256d mb = _mm256_loadu_pd(&b[i]);
    const __m256d mc = _mm256_sub_pd(ma, mb);
    _mm256_storeu_pd(&c[i], mc);
  }
  for (; i < len; i++) {
    c[i] = a[i] - b[i];
  }
  return dc;
}

static inline void darray_print_spaces(int n_spaces, FILE *out) {
  for (int i = 0; i < n_spaces; i++) {
    fprintf(out, " ");
  }
}

static inline void darray_print_spaced_dots(int n_chars, FILE *out) {
  int n_spaces = (n_chars - 4) / 2;
  darray_print_spaces(n_spaces, out);
  fwprintf(out, L"%lc", 0x00002026);
  darray_print_spaces(n_spaces, out);
}

static inline void darray_print_part_of_row(const darray *da,
                                            const char float_format[],
                                            int start, int end, int row,
                                            FILE *out) {
  ssize_t j = start;
  for (; j < end; j++) {
    fprintf(out, float_format, darray_get_elem_no_checks(da, row, j));
  }
}

static inline void darray_print_truncated_row(const darray *da,
                                              const char float_format[],
                                              int n_chars, int row, FILE *out) {
  const ssize_t n_elems = (MAX_PRINT_WIDTH >> 1) / n_chars;
  darray_print_part_of_row(da, float_format, 0, n_elems, row, out);
  darray_print_spaced_dots(n_chars, out);
  darray_print_part_of_row(da, float_format, da->dims.n_cols - n_elems,
                           da->dims.n_cols, row, out);
}

static inline void darray_print_first_row(darray *da, FILE *out,
                                          const char *float_format,
                                          int n_chars) {
  fprintf(out, "darray([[");
  if (da->dims.n_rows > 0) {
    if (n_chars * da->dims.n_cols >= MAX_PRINT_WIDTH) {
      darray_print_truncated_row(da, float_format, n_chars, 0, out);
    } else {
      darray_print_part_of_row(da, float_format, 0, da->dims.n_cols, 0, out);
    }
  }
  fprintf(out, "]\n");
}

static inline void darray_print_rows_from(darray *da, ssize_t start,
                                          ssize_t end,
                                          const char float_format[],
                                          int n_chars, FILE *out) {
  for (ssize_t i = start; i < end; i++) {
    fprintf(out, "        [");
    if (n_chars * da->dims.n_cols >= MAX_PRINT_WIDTH) {
      darray_print_truncated_row(da, float_format, n_chars, i, out);
    } else {
      darray_print_part_of_row(da, float_format, 0, da->dims.n_cols, i, out);
    }
    fprintf(out, "]\n");
  }
}

__attribute__((constructor)) static void init_locale() {
  setlocale(LC_CTYPE, "");
}

static inline void print_vertical_dots(int n_chars, FILE *out) {
  const ssize_t n_elems = (MAX_PRINT_WIDTH >> 1) / n_chars;
  const wchar_t vertical_dots = 0x000022EE;
  for (int i = 0; i < 2; i++) {
    darray_print_spaces(2 + n_chars / 2, out);
    for (int j = 0; j < n_elems; j++) {
      darray_print_spaces(n_chars, out);
      fwprintf(out, L"%lc", vertical_dots);
    }
    darray_print_spaces(2 + n_chars / 2, out);
    for (int j = 0; j < n_elems; j++) {
      darray_print_spaces(n_chars, out);
      fwprintf(out, L"%lc", vertical_dots);
    }
    fprintf(out, "\n");
  }
}

static inline ssize_t darray_print_mid_rows(darray *da, FILE *out,
                                            const char *float_format,
                                            int n_chars) {
  if (da->dims.n_rows > MAX_PRINT_HEIGHT) {
    darray_print_rows_from(da, 1, 3, float_format, n_chars, out);
    print_vertical_dots(n_chars, out);
    darray_print_rows_from(da, da->dims.n_rows - 4, da->dims.n_rows - 1,
                           float_format, n_chars, out);
    return da->dims.n_rows - 1;
  } else {
    darray_print_rows_from(da, 1, da->dims.n_rows - 1, float_format, n_chars,
                           out);
    return da->dims.n_rows - 1;
  }
}
static inline void darray_print_last_row(darray *da, FILE *out,
                                         const char *float_format,
                                         int n_chars) {
  fprintf(out, "        [");
  if (n_chars * da->dims.n_cols >= MAX_PRINT_WIDTH) {
    darray_print_truncated_row(da, float_format, n_chars, da->dims.n_rows - 1,
                               out);
  } else {
    darray_print_part_of_row(da, float_format, 0, da->dims.n_cols,
                             da->dims.n_rows - 1, out);
  }
  fprintf(out, "]])\n");
}

void cubic(const darray *da, const darray *db, darray *dc) {
  for (int i = 0; i < da->dims.n_rows; i++) {
    for (int j = 0; j < da->dims.n_cols; j++) {
      double temp = .0;
      for (int k = 0; k < db->dims.n_cols; k++) {
        temp += darray_get_elem_no_checks(da, i, k) *
                darray_get_elem_no_checks(db, k, j);
      }
      darray_set_elem_no_checks(dc, i, j, temp);
    }
  }
}

void darray_view_cubic_matmul(const darray_view *dva, const darray_view *dvb,
                              const darray_view *dvc) {
  const ssize_t n_rows_A = dva->row_end - dva->row_start;
  const ssize_t n_cols_A = dva->col_end - dva->col_start;
  const ssize_t n_cols_B = dvb->col_end - dvb->col_start;

  for (int i = 0; i < n_rows_A; i++) {
    for (int j = 0; j < n_cols_A; j++) {
      double temp = .0;
      for (int k = 0; k < n_cols_B; k++) {
        temp += darray_view_get_elem_no_checks(dva, i, k) *
                darray_view_get_elem_no_checks(dvb, k, j);
      }
      darray_view_set_elem_no_checks(
          dvc, i, j, temp + darray_view_get_elem_no_checks(dvc, i, j));
    }
  }
}

darray *cache_friendly_transpose(const darray *da) {
  const size_t n_rows = da->dims.n_rows;
  const size_t n_cols = da->dims.n_cols;
  darray *transposed = darray_new_uninitialized(n_cols, n_rows);

  size_t const max_i = (n_rows / 4) * 4;
  size_t const max_j = (n_cols / 4) * 4;

  for (size_t i = 0; i != max_i; i += 4) {
    for (size_t j = 0; j != max_j; j += 4)
      for (size_t k = 0; k != 4; ++k)
        for (size_t l = 0; l != 4; ++l)
          darray_set_elem_no_checks(
              transposed, j + l, i + k,
              darray_get_elem_no_checks(da, i + k, j + l));

    for (size_t k = 0; k != 4; ++k)
      for (size_t j = max_j; j < n_cols; ++j)
        darray_set_elem_no_checks(transposed, j, i + k,
                                  darray_get_elem_no_checks(da, i + k, j));
  }

  for (size_t i = max_i; i != n_rows; ++i)
    for (size_t j = 0; j != n_cols; ++j)
      darray_set_elem_no_checks(transposed, j, i,
                                darray_get_elem_no_checks(da, i, j));

  return transposed;
}

darray *sse_2pack_transpose(const darray *da) {
  darray *transposed =
      darray_new_uninitialized(da->dims.n_cols, da->dims.n_rows);
  __m128d r0, r1;

  const size_t n_rows = da->dims.n_rows;
  const size_t n_cols = da->dims.n_cols;

  // round to n_rows and n_cols to an even number
  const size_t max_i = (n_rows / 2) * 2;
  const size_t max_j = (n_cols / 2) * 2;

  // if the number of cols was even
  if (max_j != n_cols)
    for (size_t i = 0; i < max_i; i += 2) {
      for (size_t j = 0; j < max_j; j += 2) {
        const double *d0 = darray_get_elem_ptr_no_checks(da, i, j);
        const double *d1 = darray_get_elem_ptr_no_checks(da, i + 1, j);
        r0 = IS_ALIGNED(d0, 16) ? _mm_load_pd(d0) : _mm_loadu_pd(d0);
        r1 = IS_ALIGNED(d1, 16) ? _mm_load_pd(d1) : _mm_loadu_pd(d1);

        _mm_store_pd(darray_get_elem_ptr_no_checks(transposed, j, i),
                     _mm_shuffle_pd(r0, r1, 0x0));

        _mm_store_pd(darray_get_elem_ptr_no_checks(transposed, j + 1, i),
                     _mm_shuffle_pd(r0, r1, 0x3));
      }

      for (size_t k = 0; k < 2; ++k)
        for (size_t j = max_j; j < n_cols; ++j)
          darray_set_elem_no_checks(transposed, j, i + k,
                                    darray_get_elem_no_checks(da, i + k, j));
    }
  else
    for (size_t i = 0; i < max_i; i += 2) {
      for (size_t j = 0; j < max_j; j += 2) {
        const double *d0 = darray_get_elem_ptr_no_checks(da, i, j);
        const double *d1 = darray_get_elem_ptr_no_checks(da, i + 1, j);
        r0 = IS_ALIGNED(d0, 16) ? _mm_load_pd(d0) : _mm_loadu_pd(d0);
        r1 = IS_ALIGNED(d1, 16) ? _mm_load_pd(d1) : _mm_loadu_pd(d1);

        _mm_store_pd(darray_get_elem_ptr_no_checks(transposed, j, i),
                     _mm_shuffle_pd(r0, r1, 0x0));

        _mm_store_pd(darray_get_elem_ptr_no_checks(transposed, j + 1, i),
                     _mm_shuffle_pd(r0, r1, 0x3));
      }
    }

  if (max_i != n_rows) {
    for (size_t j = 0; j < n_cols; ++j) {
      darray_set_elem_no_checks(transposed, j, max_i,
                                darray_get_elem_no_checks(da, max_i, j));
    }
  }

  return transposed;
}

darray *darray_transpose(const darray *da) {
  return cache_friendly_transpose(da);
}

static void darray_pad_rows(darray *da, darray *da_copy) {
  darray_copy_into(da, da_copy);
  ssize_t n_skip = (da->dims.n_rows * da_copy->dims.n_cols);
  /* set the remaining elements to zero */
  memset(da_copy->elems + n_skip, '\0',
         (darray_get_size(da_copy) - n_skip) * sizeof(double));
}

static void darray_strip_rows(darray *da, darray *da_copy) {
  // assuming the only thing that's different is the number of rows, we use
  // da_copy->dims.n_rows
  double *s = da->elems;
  double *t = da_copy->elems;
  ssize_t n_cols = MIN(da->dims.n_cols, da_copy->dims.n_cols);
  for (ssize_t i = 0; i < da_copy->dims.n_rows;
       i++, t += da_copy->dims.n_cols, s += da->dims.n_cols) {
    memcpy(t, s, n_cols * sizeof(double));
  }
}

static void darray_strip_cols(darray *da, darray *da_copy) {
  double *s = da->elems;
  double *t = da_copy->elems;
  ssize_t n_rows = MIN(da->dims.n_rows, da_copy->dims.n_rows);
  for (ssize_t i = 0; i < n_rows;
       i++, t += da_copy->dims.n_cols, s += da->dims.n_cols) {
    memcpy(t, s, da_copy->dims.n_cols * sizeof(double));
  }
}

static void darray_pad_cols(darray *da, darray *da_copy) {
  double *s = da_copy->elems + da->dims.n_cols;
  for (ssize_t i = 0; i < da->dims.n_rows; i++, s += da_copy->dims.n_cols) {
    memset(s, '\0', (da_copy->dims.n_cols - da->dims.n_cols) * sizeof(double));
  }
}

darray *darray_pad(darray *da, dims new_dims) {
  exit_if_null(da, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  exit_if_new_dims_smaller(da->dims, new_dims, "must pad to larger size",
                           __LINE__);
  ssize_t ncols_new = MAX(new_dims.n_cols, da->dims.n_cols);
  ssize_t nrows_new = MAX(new_dims.n_rows, da->dims.n_rows);
  if (ncols_new == da->dims.n_cols && nrows_new == da->dims.n_rows) {
    /* nothing to pad */
    return darray_copy(da);
  }
  darray *da_copy = darray_new_uninitialized(nrows_new, ncols_new);
  darray_copy_into(da, da_copy);
  if (ncols_new == da->dims.n_cols && nrows_new != da->dims.n_rows) {
    /* pad only the rows */
    darray_pad_rows(da, da_copy);
  } else if (ncols_new != da->dims.n_cols && nrows_new == da->dims.n_rows) {
    /* pad only the columns */
    darray_pad_cols(da, da_copy);
  } else {
    /* pad both the columns and rows*/
    darray_pad_rows(da, da_copy);
    darray_pad_cols(da, da_copy);
  }
  return da_copy;
}

darray *darray_strip(darray *da, dims new_dims) {
  exit_if_null(da, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  exit_if_new_dims_greater(da->dims, new_dims,
                           "need new dims to be smaller for stripping",
                           __LINE__);

  ssize_t ncols_new = MIN(new_dims.n_cols, da->dims.n_cols);
  ssize_t nrows_new = MIN(new_dims.n_rows, da->dims.n_rows);

  darray *da_copy = darray_new_uninitialized(nrows_new, ncols_new);
  if (ncols_new == da->dims.n_cols && nrows_new != da->dims.n_rows) {
    /* strip only the rows */
    darray_strip_rows(da, da_copy);
  } else if (ncols_new != da->dims.n_cols && nrows_new == da->dims.n_rows) {
    /* strip only the columns */
    darray_strip_cols(da, da_copy);
  } else {
    /* strip both the columns and rows*/
    darray_strip_rows(da, da_copy);
    darray_strip_cols(da, da_copy);
  }
  return da_copy;
}

void darray_partition(darray *d, darray_view **dv00, darray_view **dv01,
                      darray_view **dv10, darray_view **dv11) {
  const ssize_t n_rows = d->dims.n_rows;
  const ssize_t n_cols = d->dims.n_cols;
  const ssize_t half_n_rows = n_rows / 2;
  const ssize_t half_n_cols = n_cols / 2;

  *dv00 = darray_view_new(d, 0, half_n_rows, 0, half_n_cols);
  *dv01 = darray_view_new(d, 0, half_n_rows, half_n_cols, n_cols);
  *dv10 = darray_view_new(d, half_n_rows, n_rows, 0, half_n_cols);
  *dv11 = darray_view_new(d, half_n_rows, n_rows, half_n_cols, n_cols);
}

darray_view *darray_view_from_darray(darray *d) {
  return darray_view_new(d, 0, d->dims.n_rows, 0, d->dims.n_cols);
}

void darray_view_partition(const darray_view *dv, darray_view **dv00,
                           darray_view **dv01, darray_view **dv10,
                           darray_view **dv11) {
  const ssize_t row_start = dv->row_start;
  const ssize_t row_end = dv->row_end;
  const ssize_t col_start = dv->col_start;
  const ssize_t col_end = dv->col_end;
  const ssize_t n_rows = row_end - row_start;
  const ssize_t n_cols = col_end - col_start;
  const ssize_t half_n_rows = n_rows / 2;
  const ssize_t half_n_cols = n_cols / 2;

  *dv00 = darray_view_new(dv->d, row_start, row_start + half_n_rows, col_start,
                          col_start + half_n_cols);
  *dv01 = darray_view_new(dv->d, row_start, row_start + half_n_rows,
                          col_start + half_n_cols, col_start + n_cols);
  *dv10 = darray_view_new(dv->d, row_start + half_n_rows, row_start + n_rows,
                          col_start, col_start + half_n_cols);
  *dv11 = darray_view_new(dv->d, row_start + half_n_rows, row_start + n_rows,
                          col_start + half_n_cols, col_start + n_cols);
}

darray *darray_from_view(darray_view *dv) {
  ssize_t n_rows = dv->row_end - dv->row_start;
  ssize_t n_col_elems = dv->col_end - dv->col_start;
  darray *d = darray_new_uninitialized(n_rows, n_col_elems);
  for (ssize_t i = 0; i < n_rows; i++) {
    memcpy(d->elems + (i * n_col_elems),
           dv->d->elems +
               ((i + dv->row_start) * dv->d->dims.n_cols + dv->col_start),
           n_col_elems * sizeof(double));
  }
  return d;
}

// static inline ssize_t next_power_of_two(ssize_t val) {}

#define ROW_SIZE_THRESHOLD (10)

void strassen_impl(darray_view *A, darray_view *B, darray_view *C,
                   int cur_depth, int max_depth) {
  if (cur_depth >= max_depth || A->d->dims.n_rows <= ROW_SIZE_THRESHOLD) {
    darray_view_multiply(A, B, C);
    return;
  }

  darray_view *A00, *A01, *A10, *A11, *B00, *B01, *B10, *B11, *C00, *C01, *C10,
      *C11;
  darray_view_partition(A, &A00, &A01, &A10, &A11);
  darray_view_partition(B, &B00, &B01, &B10, &B11);
  darray_view_partition(C, &C00, &C01, &C10, &C11);

  darray *S1 = darray_view_sub(B01, B11);
  darray *S2 = darray_view_add(A00, A01);
}

// assumes everything is okay
// void strassen(darray *da, darray *db) { if () }

static inline double darray_view_get_elem(darray_view *dv, ssize_t row_num,
                                          ssize_t col_num) {
  const ssize_t n_rows_view = dv->row_end - dv->row_start;
  const ssize_t n_cols_view = dv->col_end - dv->col_start;
  exit_if_violates_bounds(row_num, n_rows_view, __LINE__,
                          "row index violates bounds");
  exit_if_violates_bounds(col_num, n_cols_view, __LINE__,
                          "column index violates bounds");
  return dv->d->elems[((row_num + dv->row_start) * dv->d->dims.n_cols) +
                      col_num + dv->col_start];
}

static inline double darray_view_set_elem(darray_view *dv, ssize_t row_num,
                                          ssize_t col_num, double elem) {
  const ssize_t n_rows_view = dv->row_end - dv->row_start;
  const ssize_t n_cols_view = dv->col_end - dv->col_start;
  exit_if_violates_bounds(row_num, n_rows_view, __LINE__,
                          "row index violates bounds");
  exit_if_violates_bounds(col_num, n_cols_view, __LINE__,
                          "column index violates bounds");
  return dv->d->elems[((row_num + dv->row_start) * dv->d->dims.n_cols) +
                      col_num + dv->col_start] = elem;
}

void darray_view_multiply(darray_view *dva, darray_view *dvb,
                          darray_view *dvc) {
  exit_if_null(dva, __LINE__, "first matrix is null");
  exit_if_null(dvb, __LINE__, "second matrix is null");
  dims dims_c =
      exit_if_matrix_multiply_dims_mismatch(dva->d->dims, dvb->d->dims);

  // for each row of A
  const ssize_t n_rows_view_A = dva->row_end - dva->row_start;
  const ssize_t n_cols_view_A = dva->col_end - dva->col_start;
  const ssize_t n_cols_view_B = dvb->col_end - dvb->col_start;

  ssize_t i, j, k;
  // for each element in a row of A
  for (i = 0; i < n_rows_view_A; i++) {
    // for each element of a row of A
    for (j = 0; j < n_cols_view_A; j++) {
      double temp = .0;
      for (k = 0; k < n_cols_view_B; k++) {
        // for each element in a column of B
        const double A_ik = darray_view_get_elem_no_checks(dva, i, k);
        const double B_kj = darray_view_get_elem_no_checks(dvb, k, j);
        temp += (A_ik * B_kj);
      }
      const double C_ij = darray_view_get_elem_no_checks(dvc, i, j);
      darray_view_set_elem_no_checks(dvc, i, j, temp + C_ij);
    }
  }
}

#define MIN_THRESHOLD (1)
#define MAX_THRESHOLD (64)

void vectorized_cubic(const darray_view *dva, const darray_view *dvb,
                      const darray_view *dvc) {
  // base case
  const dims dims_dva = darray_view_get_dims(dva);
  if (MIN(dims_dva.n_cols, dims_dva.n_rows) <= MIN_THRESHOLD) {
    darray_view_cubic_matmul(dva, dvb, dvc);
    return;
  }

  const dims dims_temp = darray_view_get_dims(dvc);

  darray *dt d_autofree = darray_new_zeroed(dims_temp.n_rows, dims_temp.n_cols);
  darray_view *dvt dv_autofree = darray_view_from_darray(dt);

  darray_view *A00 dv_autofree, *A01 dv_autofree, *A10 dv_autofree,
      *A11 dv_autofree;
  darray_view *B00 dv_autofree, *B01 dv_autofree, *B10 dv_autofree,
      *B11 dv_autofree;
  darray_view *C00 dv_autofree, *C01 dv_autofree, *C10 dv_autofree,
      *C11 dv_autofree;
  darray_view *T00 dv_autofree, *T01 dv_autofree, *T10 dv_autofree,
      *T11 dv_autofree;

  darray_view_partition(dva, &A00, &A01, &A10, &A11);
  darray_view_partition(dvb, &B00, &B01, &B10, &B11);
  darray_view_partition(dvc, &C00, &C01, &C10, &C11);
  darray_view_partition(dvt, &T00, &T01, &T10, &T11);

  assert((A00->row_end - A00->row_start + A10->row_end - A11->row_start) ==
         (dva->row_end - dva->row_start));
  assert((B00->row_end - B00->row_start + B10->row_end - B11->row_start) ==
         (dvb->row_end - dvb->row_start));
  assert((C00->row_end - C00->row_start + C10->row_end - C11->row_start) ==
         (dvc->row_end - dvc->row_start));
  assert((T00->row_end - T00->row_start + T10->row_end - T11->row_start) ==
         (dvt->row_end - dvt->row_start));
  assert((A00->col_end - A00->col_start + A01->col_end - A01->col_start) ==
         (dva->col_end - dva->col_start));
  assert((B00->col_end - B00->col_start + B01->col_end - B01->col_start) ==
         (dvb->col_end - dvb->col_start));
  assert((C00->col_end - C00->col_start + C01->col_end - C01->col_start) ==
         (dvc->col_end - dvc->col_start));
  assert((T00->col_end - T00->col_start + T01->col_end - T01->col_start) ==
         (dvt->col_end - dvt->col_start));

  assert((A01->row_end - A01->row_start + A11->row_end - A11->row_start) ==
         (dva->row_end - dva->row_start));
  assert((B01->row_end - B01->row_start + B11->row_end - B11->row_start) ==
         (dvb->row_end - dvb->row_start));
  assert((C01->row_end - C01->row_start + C11->row_end - C11->row_start) ==
         (dvc->row_end - dvc->row_start));
  assert((T01->row_end - T01->row_start + T10->row_end - T11->row_start) ==
         (dvt->row_end - dvt->row_start));
  assert((A10->col_end - A10->col_start + A11->col_end - A11->col_start) ==
         (dva->col_end - dva->col_start));
  assert((B10->col_end - B10->col_start + B11->col_end - B11->col_start) ==
         (dvb->col_end - dvb->col_start));
  assert((C10->col_end - C10->col_start + C11->col_end - C11->col_start) ==
         (dvc->col_end - dvc->col_start));
  assert((T10->col_end - T10->col_start + T11->col_end - T11->col_start) ==
         (dvt->col_end - dvt->col_start));

#pragma omp parallel default(none) shared(A00, B00) private(C00)
  vectorized_cubic(A00, B00, C00);
#pragma omp parallel default(none) shared(A00, B01) private(C01)
  vectorized_cubic(A00, B01, C01);
#pragma omp parallel default(none) shared(A10, B00) private(C10)
  vectorized_cubic(A10, B00, C10);
#pragma omp parallel default(none) shared(A10, B01) private(C11)
  vectorized_cubic(A10, B01, C11);
#pragma omp parallel default(none) shared(A01, B10) private(T00)
  vectorized_cubic(A01, B10, T00);
#pragma omp parallel default(none) shared(A01, B11) private(T01)
  vectorized_cubic(A01, B11, T01);
#pragma omp parallel default(none) shared(A11, B10) private(T10)
  vectorized_cubic(A11, B10, T10);

  vectorized_cubic(A11, B11, T11);
#pragma omp barrier

  darray_view_add_into(dvc, dvt, dvc);
}

darray *darray_multiply(darray *da, darray *db) {
  exit_if_null(da, __LINE__, "first matrix is null");
  exit_if_null(db, __LINE__, "second matrix is null");
  dims dims_c = exit_if_matrix_multiply_dims_mismatch(da->dims, db->dims);
  darray_view *dva dv_autofree =
      darray_view_new(da, 0, da->dims.n_rows, 0, da->dims.n_cols);
  darray_view *dvb dv_autofree =
      darray_view_new(db, 0, db->dims.n_rows, 0, db->dims.n_cols);
  darray *dc = darray_new_uninitialized(dims_c.n_rows, dims_c.n_cols);
  darray_view *dvc =
      darray_view_new(dc, 0, dc->dims.n_rows, 0, dc->dims.n_cols);
  darray_view_multiply(dva, dvb, dvc);
  return dvc->d;
}

static inline void darray_print_empty(FILE *out) {
  fprintf(out, "darray([[]])\n");
}

static inline int get_format_string(darray *da, char *buf, ssize_t buflen) {
  const int n_dec = 2;
  const int n_digits =
      get_number_of_digits(
          trunc(MAX(fabs(darray_find_max(da)), fabs(darray_find_min(da))))) +
      n_dec + 2;
  snprintf(buf, buflen, "%%%d.%df ", n_digits, n_dec);
  return n_digits;
}

void darray_print(darray *da, FILE *out) {
  exit_if_null(da, __LINE__, ERR_MSG_POINTER_TO_DARRAY_IS_NULL);
  exit_if_null(da, __LINE__, "null pointer passed for out file");

  if (darray_is_empty(da)) {
    darray_print_empty(out);
    return;
  }

  char fmt_str[MAX_NCHARS];
  const int n_chars = get_format_string(da, fmt_str, MAX_NCHARS);
  darray_print_first_row(da, out, fmt_str, n_chars);
  ssize_t up_to = darray_print_mid_rows(da, out, fmt_str, n_chars);
  if (up_to != 0) {
    darray_print_last_row(da, out, fmt_str, n_chars);
  }
}

void darray_view_print(darray_view *dv, FILE *out) {
  exit_if_null(dv, __LINE__, "null pointer to darray view");
  fprintf(out, "view{array = %p row = %zu : %zu, col = %zu : %zu}\n",
          (void *)dv->d, dv->row_start, dv->row_end, dv->col_start,
          dv->col_end);
}

void darray_free_wrapper(darray **da) {
  darray_free(*da);
#ifdef VERBOSE
  fprintf(stderr, GET_SUCCESS_TEXT("freed"));
  fprintf(stderr, ": %p\n", (void *)*da);
#endif
}

void darray_view_free_wrapper(darray_view **dv) {
  darray_view_free(*dv);
#ifdef VERBOSE
  fprintf(stderr, GET_SUCCESS_TEXT("freed"));
  fprintf(stderr, ": %p\n", (void *)*dv);
#endif
}

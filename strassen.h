#ifndef STRASSEN_STRASSEN_H_
#define STRASSEN_STRASSEN_H_

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#define MAX_ELEMS (unsigned int)(1 << 31)

#define REPORT_ERROR_AND_EXIT(EXP, LINE_NUM, FILE_NAME, ...)                   \
  do {                                                                         \
    fprintf(stderr, "%s:%d ", FILE_NAME, LINE_NUM);                            \
    fprintf(stderr, GET_FAILURE_TEXT("error: "));                              \
    fprintf(stderr, "" #EXP " \n");                                            \
    fprintf(stderr, __VA_ARGS__);                                              \
    fprintf(stderr, "\n");                                                     \
    exit(EXIT_FAILURE);                                                        \
  } while (0)

#define LEN_MANTISSA (4)
#define MAX_PRINT_WIDTH (100)
#define MAX_PRINT_HEIGHT (6)

typedef struct {
  ssize_t n_rows, n_cols;
} dims;

typedef struct {
  double *elems;
  dims dims;
} darray;

typedef struct {
  darray *d;
  ssize_t row_start, row_end;
  ssize_t col_start, col_end;
} darray_view;

darray *darray_new_uninitialized(ssize_t n_rows, ssize_t n_cols);

darray *darray_new_zeroed(ssize_t n_rows, ssize_t n_cols);

darray *darray_new_initialized_from_doubles(ssize_t n_rows, ssize_t n_cols,
                                            double **ds);

bool darray_equal(darray *da_this, darray *da_that);

darray *darray_copy(darray *da);

void darray_free(darray *da);

ssize_t darray_get_size(const darray *array);

bool darray_is_empty(darray *da);

void darray_set_item(darray *array, ssize_t row, ssize_t col, double value);

static inline void darray_set_elem_no_checks(darray *da, ssize_t row,
                                            ssize_t col, double value);

double darray_get_item(darray *array, ssize_t row, ssize_t col);

double darray_get_elem_no_checks(const darray *array, ssize_t row, ssize_t col);

double darray_find_max(const darray *da);

double darray_find_min(const darray *da);

darray *darray_add(const darray *da, const darray *db);

darray *darray_sub(darray *da, darray *db);

darray *darray_add_into(darray *da, darray *db, darray *dc);

darray *darray_sub_into(darray *da, darray *db, darray *dc);

darray *darray_multiply(darray *da, darray *db);

void darray_print(darray *array, FILE *out);

darray *darray_pad(darray *da, dims new_dims);

darray *darray_strip(darray *da, dims new_dims);

darray_view *darray_view_new(darray *da, ssize_t row_start, ssize_t row_end,
                            ssize_t col_start, ssize_t col_end);

void darray_view_multiply(darray_view *dva, darray_view *dvb, darray_view *dvc);

darray_view *darray_view_from_darray(darray *d);

darray *darray_from_view(darray_view *dv);

void cubic(const darray *da, const darray *db, darray *dc);

void vectorized_cubic(const darray_view *dva, const darray_view *dvb, const darray_view *dvc);

darray* darray_transpose(const darray *da);

void darray_set(darray *d, double to);

void darray_free_wrapper(darray **da);

void darray_view_free_wrapper(darray_view **dv); 

darray *darray_view_add(const darray_view *dva, const darray_view *dvb);

void darray_view_add_into(const darray_view *dva, const darray_view *dvb, const darray_view *dvc);

dims darray_view_get_dims(const darray_view *dv);

void darray_view_print(darray_view *dv, FILE *out);

#define d_autofree __attribute__((__cleanup__(darray_free_wrapper)))

#define dv_autofree __attribute__((__cleanup__(darray_view_free_wrapper)))

#define DARRAY_FOR_EACH(type, val, d, func)                                    \
  for (ssize_t row = 0; row < d->dims.n_rows; row++) {                         \
    for (ssize_t col = 0; col < d->dims.n_cols; col++) {                       \
      type val = darray_get_elem_no_checks(d, row, col);                        \
      func(val);                                                               \
    }                                                                          \
  }

#endif // STRASSEN_STRASSEN_H_

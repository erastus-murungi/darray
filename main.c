#include <stdio.h>
#include <stdlib.h>

#include "strassen.h"

#define N_ROWS (80)
#define N_COLS (100)

int main() {
  // srand(6721);
  // double a[N_ROWS][N_COLS];
  // double b[N_ROWS][N_COLS];
  // for (int i = 0; i < N_ROWS; i++) {
  //   for (int j = 0; j < N_COLS; j++) {
  //     a[i][j] = (double)RAND_MAX / rand();
  //     b[i][j] = (double)RAND_MAX / rand();
  //   }
  // }
  // darray *da =
  //     darray_new_initialized_from_doubles(N_ROWS, N_COLS, (double **)a);
  // darray *db =
  //     darray_new_initialized_from_doubles(N_ROWS, N_COLS, (double **)b);
  // darray *dc = darray_add(da, db);
  // darray *dd = darray_sub(da, db);

  // darray_print(dc, stdout);
  // darray_print(dd, stdout);

  // #define NROWS_1 6
  // #define NCOLS_1 6
  //   double e[NROWS_1][NCOLS_1];

  //   for (int i = 0; i < NROWS_1; i++) {
  //     for (int j = 0; j < NCOLS_1; j++) {
  //       e[i][j] = (double)RAND_MAX / rand();
  //     }
  //   }

  //   darray *de d_autofree =
  //       darray_new_initialized_from_doubles(NROWS_1, NCOLS_1, (double **)e);

  // darray_print(de, stdout);

  // darray *df = darray_pad(
  //     de, (dims){.n_cols = NCOLS_1 + 3, .n_rows = NROWS_1});  // pad the
  //     coluns
  // darray *dg = darray_pad(
  //     de, (dims){.n_cols = NCOLS_1, .n_rows = NROWS_1 + 3});  // pad the rows
  // darray *dh =
  //     darray_pad(de, (dims){.n_cols = NCOLS_1 + 3, .n_rows = NROWS_1 + 3});
  //     // pad both

  // darray *di =
  //     darray_strip(de, (dims){.n_cols = NCOLS_1, .n_rows = NROWS_1 - 1});  //
  //     long -- strip rows
  // darray *dj d_autofree = darray_strip(
  //     de, (dims){.n_cols = NCOLS_1 - 2, .n_rows = NROWS_1});  // strip
  //     columns
  // darray *dk d_autofree = darray_strip(
  //     de, (dims){.n_cols = NCOLS_1 - 2, .n_rows = NROWS_1 - 2});  // tall

  double A[4][4] = {{1, 2, 3, 4}, 
                    {1, 2, 1, 1}, 
                    {1, 1, 3, 1}, 
                    {1, 1, 1, 4}};
  double B[4][4] = {{1, 1, 1, 1}, 
                    {1, 1, 1, 1}, 
                    {1, 1, 1, 1}, 
                    {1, 1, 1, 1}};

  darray *dA d_autofree = darray_new_initialized_from_doubles(4, 4, A);
  darray *dB d_autofree = darray_new_initialized_from_doubles(4, 4, B);
  darray *dC d_autofree = darray_new_zeroed(dA->dims.n_rows, dB->dims.n_cols);
  darray_view *dvA = darray_view_from_darray(dA);
  darray_view *dvB = darray_view_from_darray(dB);
  darray_view *dvC = darray_view_from_darray(dC);

  vectorized_cubic(dvA, dvB, dvC);

  darray_print(dvC->d, stdout);

  // darray_view dva = darray_view_new(de, 0, 1);
  // darray *dl = darray_from_view(&dva);

  // darray *dm d_autofree = darray_multiply(dj, dk);
  // darray *dn d_autofree =
  //     darray_new_zeroed(dj->dims.n_rows, dk->dims.n_cols);
  // darray *dx d_autofree =
  //     darray_new_zeroed(dj->dims.n_rows, dk->dims.n_cols);
  // cubic(dj, dk, dn);

  // darray_view *dvj = darray_view_new(dj, 0, dj->dims.n_rows, 0,
  // dj->dims.n_cols); darray_view *dvk = darray_view_new(dk, 0,
  // dk->dims.n_rows, 0, dk->dims.n_cols); darray_view *dvx =
  // darray_view_new(dx, 0, dx->dims.n_rows, 0, dx->dims.n_cols);

  // vectorized_cubic(dvj, dvk, dvx);

  // darray *dp d_autofree = darray_transpose(dj);

  // printf("equal: %s\n", darray_equal(dm, dn) ? "true" : "false");

  // darray_print(df, stdout);
  // darray_print(dg, stdout);
  // darray_print(dh, stdout);
  // darray_print(di, stdout);
  // darray_print(dj, stdout);
  // darray_print(dk, stdout);
  // darray_print(dl, stdout);
  // darray_print(dm, stdout);
  // darray_print(dn, stdout);
  // darray_print(dvx->d, stdout);
  // darray_print(dp, stdout);

  // printf("%s\n", darray_equal(dvx->d, dn) ? "true" : "false");

  // darray_print(darray_sub(dvx->d, dn), stdout);

  return EXIT_SUCCESS;
}

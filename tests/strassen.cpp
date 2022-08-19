#include <cmath>
#include <cstring>

#include "gtest/gtest.h"

extern "C" {
#include "../strassen.h"
}

class StrassenTestDarrayAdd : public testing::Test {
protected:
  void SetUp() override {
    da = darray_new_initialized_from_doubles(
        3, 3, reinterpret_cast<double **>(a_3x3));
    db = darray_new_initialized_from_doubles(
        3, 3, reinterpret_cast<double **>(b_3x3));
    db_4x4 = darray_new_initialized_from_doubles(
        4, 4, reinterpret_cast<double **>(b_4x4));
  }
  void TearDown() override {
    darray_free(da);
    darray_free(db);
    darray_free(db_4x4);
  }

  double a_3x3[3][3] = {{0.0, 1.0, 0.0}, {130.0, 0.0, 0.0}, {0.0, 0.0, 45.4}};
  double b_3x3[3][3] = {{0.0, 4.0, 0.0}, {15.0, 0.0, 0.0}, {0.0, 0.0, 45.4}};
  double expected[3][3] = {
      {0.0, 5.0, 0.0}, {145.0, 0.0, 0.0}, {0.0, 0.0, 90.8}};
  double b_4x4[4][4] = {{0.0, 1.0, 0.0, 10.0},
                    {130.0, 0.0, 0.0, 12.3},
                    {0.0, 0.0, 45.4, 6.1},
                    {10.0, 0.0, 45.4, 6.1}};
  darray *da;
  darray *db;
  darray *db_4x4;
};

TEST(StrassenTestDarrayIsEmpty, WithDimsRow0Col0) {
  darray *d d_autofree = darray_new_uninitialized(0, 0);
  ASSERT_TRUE(darray_is_empty(d));
}

TEST(StrassenTestDarrayIsEmpty, WithDimsRow1Col0) {
  darray *d d_autofree = darray_new_uninitialized(1, 0);
  ASSERT_TRUE(darray_is_empty(d));
}

TEST(StrassenTestDarrayIsEmpty, WithDimsRow0Col1) {
  darray *d d_autofree = darray_new_uninitialized(0, 1);
  ASSERT_TRUE(darray_is_empty(d));
}

TEST(StrassenTestDarrayZeroInitialization, WithTheZeroedConstructor) {
  darray *d d_autofree = darray_new_zeroed(5, 5);
  DARRAY_FOR_EACH(double, v, d, [](double val) { EXPECT_DOUBLE_EQ(val, 0.0); });
}

TEST(StrassenTestDarrayZeroInitialization, WithTheDoublesArrayConstructor) {
  double a[5][5] = {{0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0}};
  darray *d d_autofree =
      darray_new_initialized_from_doubles(5, 5, reinterpret_cast<double **>(a));
  DARRAY_FOR_EACH(double, v, d, [](double val) { EXPECT_DOUBLE_EQ(val, 0.0); });
}

TEST(StrassenTestDarrayMax, WithInfAsMax) {
  double a[5][5] = {
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, std::numeric_limits<double>::infinity(), 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0}};
  darray *d d_autofree =
      darray_new_initialized_from_doubles(5, 5, reinterpret_cast<double **>(a));
  EXPECT_DOUBLE_EQ(darray_find_max(d), std::numeric_limits<double>::infinity());
}

TEST(StrassenTestDarrayMax, WithMaxDoubleFromNumericLimits) {
  double a[5][5] = {{0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, std::numeric_limits<double>::max(), 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0}};
  darray *d d_autofree =
      darray_new_initialized_from_doubles(5, 5, reinterpret_cast<double **>(a));
  EXPECT_DOUBLE_EQ(darray_find_max(d), std::numeric_limits<double>::max());
}

TEST(StrassenTestDarrayMax, WithMaxAsNan) {
  double a[5][5] = {
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, std::numeric_limits<double>::infinity(), 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN()}};
  darray *d d_autofree =
      darray_new_initialized_from_doubles(5, 5, reinterpret_cast<double **>(a));
  EXPECT_TRUE(std::isnan(darray_find_max(d)));
}

TEST(STrassenTestDarrayMin, WithMinusInfAsMin) {
  double a[5][5] = {
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, -std::numeric_limits<double>::infinity(), 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0}};
  darray *d d_autofree =
      darray_new_initialized_from_doubles(5, 5, reinterpret_cast<double **>(a));
  EXPECT_DOUBLE_EQ(darray_find_min(d),
                   -std::numeric_limits<double>::infinity());
}

TEST(STrassenTestDarrayMin, WithMinDoubleFromNumericLimits) {
  double a[5][5] = {{0.0, 1.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, std::numeric_limits<double>::min(), 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0, 0.0, 0.0}};
  darray *d d_autofree =
      darray_new_initialized_from_doubles(5, 5, reinterpret_cast<double **>(a));
  EXPECT_DOUBLE_EQ(darray_find_min(d), 0.0);
}

TEST_F(StrassenTestDarrayAdd, WithValidInputs) {
  darray *d_actual d_autofree = darray_add(da, db);
  ASSERT_EQ(std::memcmp(expected, d_actual->elems, sizeof(expected)), 0);
}

using StrassenDeathTestDarrayAdd = StrassenTestDarrayAdd;

TEST_F(StrassenDeathTestDarrayAdd, FailsWhenFirstArrayIsNull) {
  ASSERT_EXIT(darray_add(NULL, da), ::testing::ExitedWithCode(EXIT_FAILURE),
              "first matrix is null");
}

TEST_F(StrassenDeathTestDarrayAdd, FailsWhenSecondArrayIsNull) {
  ASSERT_EXIT(darray_add(da, NULL), ::testing::ExitedWithCode(EXIT_FAILURE),
              "second matrix is null");
}


TEST_F(StrassenDeathTestDarrayAdd, FailsWhenArrayDimsMismatch) {
  ASSERT_EXIT(darray_add(da, db_4x4), ::testing::ExitedWithCode(EXIT_FAILURE),
              "adding requires matrix dimensions to be equal");
}

TEST(StrassenTestDarraySub, WithValidInputs) {
  double a[3][3] = {{120.0, 1.0, 0.0},
                    {130.0, 0.0, 0.0},
                    {10.0, std::numeric_limits<double>::infinity(), 45.4}};
  double b[3][3] = {{13.0, 4.0, 0.0}, {15.0, 10.0, 0.0}, {3.4, 0.0, 45.4}};
  double expected[3][3] = {{107.0, -3.0, 0.0},
                           {115.0, -10.0, 0.0},
                           {6.6, std::numeric_limits<double>::infinity(), 0.0}};
  darray *da d_autofree =
      darray_new_initialized_from_doubles(3, 3, reinterpret_cast<double **>(a));
  darray *db d_autofree =
      darray_new_initialized_from_doubles(3, 3, reinterpret_cast<double **>(b));
  darray *d_actual d_autofree = darray_sub(da, db);
  ASSERT_EQ(std::memcmp(expected, d_actual->elems, sizeof(expected)), 0);
}

TEST(StrassenTestDarraySub, FailsWhenFirstArrayIsNull) {
  double a[3][3] = {{0.0, 1.0, 0.0}, {130.0, 0.0, 0.0}, {0.0, 0.0, 45.4}};
  darray *da d_autofree =
      darray_new_initialized_from_doubles(3, 3, reinterpret_cast<double **>(a));
  ASSERT_EXIT(darray_sub(NULL, da), ::testing::ExitedWithCode(EXIT_FAILURE),
              "first matrix is null");
}

TEST(StrassenTestDarraySub, FailsWhenSecondArrayIsNull) {
  double a[3][3] = {{0.0, 1.0, 0.0}, {130.0, 0.0, 0.0}, {0.0, 0.0, 45.4}};
  darray *da d_autofree =
      darray_new_initialized_from_doubles(3, 3, reinterpret_cast<double **>(a));
  ASSERT_EXIT(darray_sub(da, NULL), ::testing::ExitedWithCode(EXIT_FAILURE),
              "second matrix is null");
}

TEST(StrassenTestDarraySub, FailsWhenArrayDimsMismatch) {
  double a[3][3] = {{0.0, 1.0, 0.0}, {130.0, 0.0, 0.0}, {0.0, 0.0, 45.4}};
  double b[4][4] = {{0.0, 1.0, 0.0, 10.0},
                    {130.0, 0.0, 0.0, 12.3},
                    {0.0, 0.0, 45.4, 6.1},
                    {10.0, 0.0, 45.4, 6.1}};
  darray *da d_autofree =
      darray_new_initialized_from_doubles(3, 3, reinterpret_cast<double **>(a));
  darray *db d_autofree =
      darray_new_initialized_from_doubles(4, 4, reinterpret_cast<double **>(b));
  ASSERT_EXIT(darray_sub(da, db), ::testing::ExitedWithCode(EXIT_FAILURE),
              "adding requires matrix dimensions to be equal");
}

TEST(StrassenTestDarrayViewAdd, WithValidInputs) {
  double a[4][4] = {{0.0, 1.0, 0.0, 3.0},
                    {130.0, 0.0, 0.0, 2.0},
                    {0.0, 0.0, 45.4, 34.5},
                    {0.0, 0.0, 45.4, 34.5}};
  double b[4][4] = {{0.0, 1.0, 0.0, 10.0},
                    {130.0, 0.0, 0.0, 12.3},
                    {0.0, 0.0, 45.4, 6.1},
                    {10.0, 0.0, 45.4, 6.1}};
  double c[4][4] = {{0.0, 2.0, 0.0, 13.0},
                    {260.0, 0.0, 0.0, 14.3},
                    {0.0, 0.0, 90.8, 40.6},
                    {10.0, 0.0, 90.8, 40.6}};
  darray *da d_autofree =
      darray_new_initialized_from_doubles(4, 4, reinterpret_cast<double **>(a));
  darray *db d_autofree =
      darray_new_initialized_from_doubles(4, 4, reinterpret_cast<double **>(b));
  darray *dc d_autofree =
      darray_new_initialized_from_doubles(4, 4, reinterpret_cast<double **>(c));
  darray_view *dva dv_autofree =
      darray_view_new(da, 0, 2, 0, 2);
  darray_view *dvb dv_autofree =
      darray_view_new(db, 0, 2, 0, 2);
  darray_view *dvc dv_autofree =
      darray_view_new(dc, 0, 2, 0, 2);
  darray *d_actual d_autofree = darray_from_view(dvc);
  darray *d_ret d_autofree = darray_view_add(dva, dvb);
  ASSERT_TRUE(darray_equal(d_ret, d_actual));
}
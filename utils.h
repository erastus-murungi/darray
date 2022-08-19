#ifndef STRASSEN_UTILS_H_
#define STRASSEN_UTILS_H_

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#define GET_FAILURE_TEXT(s) "\x1b[31m" s "\033[m"
#define GET_SUCCESS_TEXT(s) "\x1b[32m" s "\033[m"

bool approximately_equal(double a, double b);
unsigned int get_number_of_digits(unsigned int i);

#define IS_ALIGNED(x, a) (((uintptr_t)(x) & (a - 1)) == 0)

#endif // STRASSEN_UTILS_H_

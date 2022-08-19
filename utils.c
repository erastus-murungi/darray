#include "utils.h"
#include <math.h>

static const uint32_t kMaxUlps = 4;
static const uint64_t kSignBitMask = 1UL << (64 - 1);

// Converts an integer from the sign-and-magnitude representation to
// the biased representation.  More precisely, let N be 2 to the
// power of (kBitCount - 1), an integer x is represented by the
// unsigned number x + N.
//
// For instance,
//
//   -N + 1 (the most negative number representable using
//          sign-and-magnitude) is represented by 1;
//   0      is represented by N; and
//   N - 1  (the biggest number representable using
//          sign-and-magnitude) is represented by 2N - 1.
//
// Read http://en.wikipedia.org/wiki/Signed_number_representations
// for more details on signed number representations.
static uint64_t SignAndMagnitudeToBiased(const uint64_t sam) {
  if (kSignBitMask & sam) {
    // sam represents a negative number.
    return ~sam + 1;
  } else {
    // sam represents a positive number.
    return kSignBitMask | sam;
  }
}

static uint64_t DistanceBetweenSignAndMagnitudeNumbers(const uint64_t sam1,
                                                       const uint64_t sam2) {
  const uint64_t biased1 = SignAndMagnitudeToBiased(sam1);
  const uint64_t biased2 = SignAndMagnitudeToBiased(sam2);
  return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}

// adapted from
// https://github.com/google/googletest/blob/1b18723e874b256c1e39378c6774a90701d70f7a/googletest/include/gtest/internal/gtest-internal.h#L344
bool approximately_equal(double a, double b) {
  // The IEEE standard says that any comparison operation involving a NAN must
  // return false.
  if (isnan(a) || isnan(b)) {
    return true;
  }

  const uint64_t a_bits = (uint64_t)a; // raw bits of a
  const uint64_t b_bits = (uint64_t)b; // raw bits of b

  return DistanceBetweenSignAndMagnitudeNumbers(a_bits, b_bits) <= kMaxUlps;
}

// https://stackoverflow.com/questions/1489830/efficient-way-to-determine-number-of-digits-in-an-integer
unsigned get_number_of_digits(unsigned i) {
  if (i > 0) {
    return (int)log10((double)i) + 1;
  } else if (i == 0) {
    return 1;
  } else {
    return (int)log10((double)-i) + 2;
  }
}

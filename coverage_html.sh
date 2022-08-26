#!/bin/bash

GREEN='\x1B[32;1m'
RESET='\x1B[0m' # No Color
printf "Running %s cmake with PROFILE=1 $%s" "$GREEN" "$RESET"
cmake CMakeLists.txt -DPROFILE=1
make
ctest
/usr/local/opt/llvm/bin/llvm-cov gcov ./CMakeFiles/strassen*/*.gcno
mkdir "temp"
gcovr --html-details temp/output.html
open temp/output.html
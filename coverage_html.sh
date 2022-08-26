cmake CMakeLists.txt -DPROFILE=1
make
ctest
/usr/local/opt/llvm/bin/llvm-cov gcov "./CMakeFiles/strassen*/*.gcno"
mkdir "temp"
gcovr --html-details temp/output.html
open temp/output.html
rm -rf "temp"
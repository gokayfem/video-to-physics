#!/bin/bash
# clang wrapper: Apple clang needs -Xpreprocessor -fopenmp + Homebrew libomp
OMP=${LIBOMP_PREFIX:-/opt/homebrew/opt/libomp}
args=()
link=1
for a in "$@"; do
  if [ "$a" == "-fopenmp" ]; then args+=("-Xpreprocessor" "-fopenmp"); else args+=("$a"); fi
  if [ "$a" == "-E" ] || [ "$a" == "-c" ]; then link=0; fi
done
if [ $link == 1 ]; then exec clang -std=c99 -D_DARWIN_C_SOURCE -Wno-unused-result -Wno-unused-function -I$OMP/include "${args[@]}" -L$OMP/lib -lomp
else exec clang -std=c99 -D_DARWIN_C_SOURCE -Wno-unused-result -Wno-unused-function -I$OMP/include "${args[@]}"; fi

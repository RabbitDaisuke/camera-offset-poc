#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${BUILD_ROOT:-$project_dir/.build-stereo}"
output_dir="${OUTPUT_DIR:-$project_dir/dist/native}"
opencv_version=4.13.0
opencv_sha256=1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d
mkdir -p "$build_root" "$output_dir"
source_dir="${OPENCV_SOURCE_DIR:-$build_root/opencv-$opencv_version}"
binary_dir="${OPENCV_BUILD_DIR:-$build_root/opencv-wasm}"
if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
  archive="$build_root/opencv-$opencv_version.tar.gz"
  curl --fail --location --retry 3 "https://github.com/opencv/opencv/archive/refs/tags/$opencv_version.tar.gz" -o "$archive"
  echo "$opencv_sha256  $archive" | sha256sum --check
  tar -xzf "$archive" -C "$build_root"
fi
emcmake cmake -S "$source_dir" -B "$binary_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_LIST=calib3d,objdetect -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_PTHREADS_PF=OFF -DWITH_TBB=OFF \
  -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DENABLE_PIC=OFF -DCPU_BASELINE= -DCPU_DISPATCH= \
  '-DCMAKE_C_FLAGS=-O3 -msimd128 -fexceptions' '-DCMAKE_CXX_FLAGS=-O3 -msimd128 -fexceptions'
cmake --build "$binary_dir" --parallel "${BUILD_JOBS:-2}"
em++ -std=c++17 -O3 -msimd128 -fexceptions --bind "$project_dir/native/stereo.cpp" \
  -I"$source_dir/modules/core/include" -I"$source_dir/modules/imgproc/include" \
  -I"$source_dir/modules/features2d/include" -I"$source_dir/modules/flann/include" \
  -I"$source_dir/modules/calib3d/include" -I"$source_dir/modules/objdetect/include" -I"$binary_dir" \
  "$binary_dir/lib/libopencv_objdetect.a" \
  "$binary_dir/lib/libopencv_calib3d.a" "$binary_dir/lib/libopencv_features2d.a" \
  "$binary_dir/lib/libopencv_flann.a" "$binary_dir/lib/libopencv_imgproc.a" \
  "$binary_dir/lib/libopencv_core.a" "$binary_dir/3rdparty/lib/libzlib.a" \
  -sMODULARIZE=1 -sEXPORT_NAME=StereoNativeFactory -sALLOW_MEMORY_GROWTH=1 \
  -sINITIAL_MEMORY=67108864 -sMAXIMUM_MEMORY=2147483648 \
  -sENVIRONMENT=web,worker,node -sFILESYSTEM=0 -sDISABLE_EXCEPTION_CATCHING=0 \
  -o "$output_dir/stereo-native.js"
cp "$source_dir/LICENSE" "$output_dir/OpenCV-LICENSE.txt"
(cd "$output_dir" && sha256sum stereo-native.js stereo-native.wasm > SHA256SUMS)

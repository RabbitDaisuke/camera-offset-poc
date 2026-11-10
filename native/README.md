# OpenCV stereo calibration binding
This module calls the unmodified OpenCV 4.13.0 `stereoCalibrate` implementation, including its per-view poses and errors. It does not replace AprilTag detection or implement a new calibration optimizer.
Upstream source: https://github.com/opencv/opencv/releases/tag/4.13.0
Toolchain: Emscripten 3.1.74, SIMD, no threads, exception support, growing WebAssembly memory.
The application retains the original OpenCV.js detector. This separate module exposes the stereo calibration function omitted by that tutorial build.
Object coordinates are in metres. Each input view is one physical paper visible in both cameras, with the same decoded marker IDs and corner indexes. Repeated IDs on different papers must be paired at the paper level before calibration.
Only square-pixel focal lengths are optimized by default. Both principal points are supplied from the corresponding image sizes. Distortion is zero unless the caller explicitly enables the first radial coefficient. No camera model or focal length is chosen from a filename.
GitHub Actions builds OpenCV and this binding on Ubuntu 24.04 with Emscripten 3.1.74. The workflow runs for native source changes and manual dispatch. Download `opencv-4.13.0-stereo-wasm` from the completed run; it contains the JavaScript loader, WASM binary, OpenCV license and SHA256 checksums. No Pages deployment is performed by this workflow.
To build with an activated Emscripten 3.1.74 environment, run `bash native/build.sh`. CMake, Ninja, curl and tar are required. `BUILD_ROOT`, `BUILD_JOBS` and `OUTPUT_DIR` may be set explicitly. Existing source and build trees may be reused through `OPENCV_SOURCE_DIR` and `OPENCV_BUILD_DIR`.
The workflow verifies module loading, the calibration export and exception handling. Surface reconstruction and video synchronization still require validation against actual paired recordings; a successful native build does not establish those results.

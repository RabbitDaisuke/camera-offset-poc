#include <emscripten/bind.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

using emscripten::val;

static cv::Mat matrix(const val &values, int rows, int cols) {
    if (values["length"].as<int>() != rows * cols)
        throw std::invalid_argument("Invalid matrix size");
    cv::Mat m(rows, cols, CV_64F);
    for (int n = 0; n < rows * cols; ++n)
        m.ptr<double>()[n] = values[n].as<double>();
    return m;
}

static val array(const cv::Mat &source) {
    cv::Mat m; source.convertTo(m, CV_64F);
    val result = val::array();
    for (size_t n = 0; n < m.total() * m.channels(); ++n)
        result.set(n, m.ptr<double>()[n]);
    return result;
}

// Thin binding to the unmodified OpenCV implementation. Image coordinates may
// come from different image sizes: USE_INTRINSIC_GUESS supplies both K matrices.
static val calibrateStereo(const val &views, const val &options) {
    try {
        std::vector<std::vector<cv::Point3f>> object;
        std::vector<std::vector<cv::Point2f>> left, right;
        const int count = views["length"].as<int>();
        if (count < 2) throw std::invalid_argument("At least two paper pairs are required");
        for (int i = 0; i < count; ++i) {
            const val view = views[i];
            const int n = view["object"]["length"].as<int>();
            if (n < 8 || view["left"]["length"].as<int>() != n || view["right"]["length"].as<int>() != n)
                throw std::invalid_argument("Corresponding paper corner counts differ");
            object.emplace_back(); left.emplace_back(); right.emplace_back();
            for (int j = 0; j < n; ++j) {
                const val p = view["object"][j], a = view["left"][j], b = view["right"][j];
                object.back().emplace_back(p[0].as<float>(), p[1].as<float>(), p[2].as<float>());
                left.back().emplace_back(a[0].as<float>(), a[1].as<float>());
                right.back().emplace_back(b[0].as<float>(), b[1].as<float>());
            }
        }
        cv::Mat K0 = matrix(options["K0"], 3, 3), K1 = matrix(options["K1"], 3, 3);
        cv::Mat D0 = options["D0"].isUndefined() ? cv::Mat::zeros(5, 1, CV_64F) : matrix(options["D0"], 5, 1);
        cv::Mat D1 = options["D1"].isUndefined() ? cv::Mat::zeros(5, 1, CV_64F) : matrix(options["D1"], 5, 1);
        cv::Mat R, T, E, F, errors;
        std::vector<cv::Mat> rvecs, tvecs;
        int flags = cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_FIX_ASPECT_RATIO |
                    cv::CALIB_FIX_PRINCIPAL_POINT | cv::CALIB_ZERO_TANGENT_DIST |
                    cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_USE_LU;
        if (!options["radial"].as<bool>()) flags |= cv::CALIB_FIX_K1;
        if (!options["R"].isUndefined() && !options["T"].isUndefined()) {
            R = matrix(options["R"], 3, 3); T = matrix(options["T"], 3, 1);
            flags |= cv::CALIB_USE_EXTRINSIC_GUESS;
        }
        const double rms = cv::stereoCalibrate(
            object, left, right, K0, D0, K1, D1,
            cv::Size(options["width"].as<int>(), options["height"].as<int>()),
            R, T, E, F, rvecs, tvecs, errors, flags,
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 500, 1e-10));
        val result = val::object(), poses = val::array();
        result.set("rms", rms); result.set("K0", array(K0)); result.set("K1", array(K1));
        result.set("D0", array(D0)); result.set("D1", array(D1));
        result.set("R", array(R)); result.set("T", array(T));
        result.set("E", array(E)); result.set("F", array(F));
        result.set("viewErrors", array(errors)); result.set("flags", flags);
        result.set("version", CV_VERSION);
        for (size_t i = 0; i < rvecs.size(); ++i) {
            cv::Mat rotation; cv::Rodrigues(rvecs[i], rotation);
            val pose = val::object(); pose.set("R", array(rotation)); pose.set("t", array(tvecs[i]));
            poses.set(i, pose);
        }
        result.set("poses", poses);
        return result;
    } catch (const std::exception &e) {
        val error = val::object(); error.set("error", std::string(e.what())); return error;
    }
}

static val undistortPixels(const val &points, const val &camera) {
    std::vector<cv::Point2d> input, output;
    for (int i = 0; i < points["length"].as<int>(); ++i)
        input.emplace_back(points[i][0].as<double>(), points[i][1].as<double>());
    cv::undistortPoints(input, output, matrix(camera["K"], 3, 3), matrix(camera["D"], 5, 1));
    val result = val::array();
    for (size_t i = 0; i < output.size(); ++i) {
        val p = val::array(); p.set(0, output[i].x); p.set(1, output[i].y); p.set(2, 1.0);
        result.set(i, p);
    }
    return result;
}

static val stereoDisparity(const val &left, const val &right, int width, int height,
                          int minimum, int disparities) {
    cv::Mat a(height, width, CV_8UC3), b(height, width, CV_8UC3), da, db;
    if (left["length"].as<int>() != width * height * 3 || right["length"].as<int>() != width * height * 3)
        throw std::invalid_argument("RGB image dimensions differ");
    val(emscripten::typed_memory_view(a.total() * 3, a.data)).call<void>("set", left);
    val(emscripten::typed_memory_view(b.total() * 3, b.data)).call<void>("set", right);
    disparities = ((disparities + 15) / 16) * 16;
    const int block = 5, rightMinimum = -minimum - disparities + 1;
    // SGBM reserves the full disparity range at both image edges. Padding
    // retains the real overlap when the requested depth range is wide;
    // downstream RGB/valid-image checks reject padding correspondences.
    const int pad = minimum + disparities + block;
    cv::Mat paddedA = cv::Mat::zeros(height, width + 2 * pad, CV_8UC3), paddedB = paddedA.clone();
    const cv::Rect original(pad, 0, width, height);
    a.copyTo(paddedA(original)); b.copyTo(paddedB(original));
    auto matcher = cv::StereoSGBM::create(minimum, disparities, block, 8 * 3 * block * block,
        32 * 3 * block * block, 1, 31, 12, 60, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
    matcher->compute(paddedA, paddedB, da);
    matcher->setMinDisparity(rightMinimum); matcher->compute(paddedB, paddedA, db);
    da = da(original).clone(); db = db(original).clone();
    val result = val::object();
    result.set("left", val(emscripten::typed_memory_view(da.total(), da.ptr<int16_t>())).call<val>("slice"));
    result.set("right", val(emscripten::typed_memory_view(db.total(), db.ptr<int16_t>())).call<val>("slice"));
    result.set("minimum", minimum); result.set("rightMinimum", rightMinimum);
    return result;
}

// Linear postprocessing for GPU-produced 3-way SGBM winners. The GPU retains
// the same cost, uniqueness and subpixel rules; OpenCV retains its exact
// pseudo left/right check, median filter and connected-component filter.
static val finishGpuDisparity(const val &raw, int width, int height,
                             int minimum, int disparities, bool reverse) {
    disparities = ((disparities + 15) / 16) * 16;
    const int pad = minimum + disparities + 5, paddedWidth = width + 2 * pad;
    const int md = reverse ? -minimum - disparities + 1 : minimum;
    const int minX = std::max(md + disparities, 0);
    const int maxX = paddedWidth + std::min(md, 0), validWidth = maxX - minX;
    if (raw["length"].as<int>() != height * validWidth * 3)
        throw std::invalid_argument("GPU disparity dimensions differ");
    std::vector<int32_t> values(height * validWidth * 3);
    val(emscripten::typed_memory_view(values.size(), values.data())).call<void>("set", raw);
    const short invalid = static_cast<short>((md - 1) * 16);
    cv::Mat disparity(height, paddedWidth, CV_16S, cv::Scalar(invalid));
    std::vector<int> back(paddedWidth), costs(paddedWidth);
    for (int y = 0; y < height; ++y) {
        std::fill(back.begin(), back.end(), invalid);
        std::fill(costs.begin(), costs.end(), 32767);
        short *row = disparity.ptr<short>(y);
        for (int x = validWidth - 1; x >= 0; --x) {
            const int32_t *p = values.data() + (y * validWidth + x) * 3;
            row[x + minX] = static_cast<short>(p[0]);
            if (p[0] == invalid) continue;
            const int xr = x + minX - p[1];
            if (xr >= 0 && xr < paddedWidth && costs[xr] > p[2]) {
                costs[xr] = p[2]; back[xr] = p[1];
            }
        }
        for (int x = minX; x < maxX; ++x) {
            const int d = row[x]; if (d == invalid) continue;
            const int lo = d >> 4, hi = (d + 15) >> 4;
            const int a = x - lo, b = x - hi;
            if (a >= 0 && a < paddedWidth && back[a] >= md && std::abs(back[a]-lo) > 1 &&
                b >= 0 && b < paddedWidth && back[b] >= md && std::abs(back[b]-hi) > 1)
                row[x] = invalid;
        }
    }
    cv::medianBlur(disparity, disparity, 3);
    cv::filterSpeckles(disparity, invalid, 60, 32);
    const cv::Mat result = disparity(cv::Rect(pad, 0, width, height)).clone();
    return val(emscripten::typed_memory_view(result.total(), result.ptr<int16_t>())).call<val>("slice");
}

EMSCRIPTEN_BINDINGS(epipolar_stereo) {
    emscripten::function("calibrateStereo", &calibrateStereo);
    emscripten::function("undistortPixels", &undistortPixels);
    emscripten::function("stereoDisparity", &stereoDisparity);
    emscripten::function("finishGpuDisparity", &finishGpuDisparity);
}

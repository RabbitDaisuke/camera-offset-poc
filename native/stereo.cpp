#include <emscripten/bind.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/version.hpp>
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

EMSCRIPTEN_BINDINGS(epipolar_stereo) {
    emscripten::function("calibrateStereo", &calibrateStereo);
}

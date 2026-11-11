#include <emscripten/bind.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <stdexcept>
#include <vector>

using emscripten::val;

static cv::aruco::ArucoDetector makeDetector(bool fast) {
    cv::aruco::DetectorParameters p;
    p.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    p.cornerRefinementWinSize = 3;
    p.aprilTagQuadDecimate = fast ? 2 : 1;
    p.minMarkerPerimeterRate = .005;
    p.polygonalApproxAccuracyRate = .05;
    p.errorCorrectionRate = .45;
    p.adaptiveThreshWinSizeMax = 73;
    return cv::aruco::ArucoDetector(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11),p,cv::aruco::RefineParameters(10,3,true));
}

// The same official detector, dictionary and parameters as core.js. The
// grayscale input is unchanged; only the compiler and WASM SIMD build differ.
static val detectAprilTags(const val &gray, int width, int height, bool fast) {
    if (gray["length"].as<int>() != width*height) throw std::invalid_argument("Gray image dimensions differ");
    static auto full = makeDetector(false), roi = makeDetector(true);
    cv::Mat image(height,width,CV_8UC1);
    val(emscripten::typed_memory_view(image.total(),image.data)).call<void>("set",gray);
    std::vector<std::vector<cv::Point2f>> corners,rejected;
    std::vector<int> ids;
    (fast ? roi : full).detectMarkers(image,corners,ids,rejected);
    auto flatten=[](const std::vector<std::vector<cv::Point2f>> &values,size_t limit) {
        std::vector<float> result;
        for(size_t i=0;i<values.size() && i<limit;++i) for(const auto &p:values[i]) { result.push_back(p.x);result.push_back(p.y); }
        return val(emscripten::typed_memory_view(result.size(),result.data())).call<val>("slice");
    };
    val result=val::object();
    result.set("ids",val(emscripten::typed_memory_view(ids.size(),ids.data())).call<val>("slice"));
    result.set("corners",flatten(corners,corners.size()));
    result.set("rejected",flatten(rejected,80));
    return result;
}

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

EMSCRIPTEN_BINDINGS(epipolar_stereo) {
    emscripten::constant("aprilTagStackBytes", 5242880);
    emscripten::function("detectAprilTags", &detectAprilTags);
    emscripten::function("calibrateStereo", &calibrateStereo);
    emscripten::function("undistortPixels", &undistortPixels);
    emscripten::function("stereoDisparity", &stereoDisparity);
}

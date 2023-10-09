#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>
#include "lifting_3d.hpp"


extern std::shared_ptr<spdlog::logger> logger;

using namespace cv;
using namespace std;

const bool debug = true;

const int DELTA = 600;

const int VEHICLE_L = 4100 + DELTA;
const int VEHICLE_W = 2100 + DELTA;
const int VEHICLE_H = 1700 + DELTA;

const int MOTORCYCLE_L = 2400 + DELTA;
const int MOTORCYCLE_W = 900 + DELTA;
const int MOTORCYCLE_H = 2200 + DELTA;

const int VAN_L = 5400 + DELTA;
const int VAN_W = 2300 + DELTA;
const int VAN_H = 2300 + DELTA;

const int IMAGE_H = 14000;
const int IMAGE_W = 2500;

double pixels_in_mm;
double mm_in_pix;
double pixels_in_mm_origin;

const int max_debug_index = 300;
const int min_debug_index = 200;
const int thickness = 2;
const int H = 7000;
const int W = 700;

const int X_OFFSET = 100;
const int Y_OFFSET = 6000;

cv::Point2f get_intersect(cv::Point2f a1, cv::Point2f a2, cv::Point2f b1, cv::Point2f b2)
{
    cv::Matx33f A(a1.x - a2.x, b2.x - b1.x, a1.x - b1.x,
                  a1.y - a2.y, b2.y - b1.y, a1.y - b1.y,
                  0, 0, 1);

    cv::Vec3f B(b1.x - a2.x, b1.y - a2.y, 0);

    cv::Vec3f X = A.inv() * B;

    return cv::Point2f(X[0], X[1]);
}

cv::Point2f warp_perspective(cv::Point2f p, cv::Matx33f matrix)
{
    float px = (matrix(0, 0) * p.x + matrix(0, 1) * p.y + matrix(0, 2)) /
               (matrix(2, 0) * p.x + matrix(2, 1) * p.y + matrix(2, 2));

    float py = (matrix(1, 0) * p.x + matrix(1, 1) * p.y + matrix(1, 2)) /
               (matrix(2, 0) * p.x + matrix(2, 1) * p.y + matrix(2, 2));

    return cv::Point2f(px, py);
}

vector<cv::Point2f> warp(vector<cv::Point2f> a, int i, cv::Matx33f matrix)
{
    vector<cv::Point2f> result;
    result.push_back(warp_perspective(a[i], matrix));
    return result;
}

vector<cv::Point2f> to_warp(vector<cv::Point2f> a, cv::Matx33f matrix)
{
    vector<cv::Point2f> result;
    for (int i = 0; i < a.size(); i++)
    {
        vector<cv::Point2f> warped = warp(a, i, matrix);
        result.insert(result.end(), warped.begin(), warped.end());
    }
    return result;
}

cv::Point2f iv(cv::Point2f a)
{
    return cv::Point2f(a.x, -a.y);
}

vector<cv::Point2f> to_iv(vector<cv::Point2f> array)
{
    vector<cv::Point2f> converted;
    for (int i = 0; i < array.size(); i++)
    {
        converted.push_back(iv(array[i]));
    }
    return converted;
}

double get_angle(cv::Point2f a, cv::Point2f b, cv::Point2f c)
{
    double ang = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x);
    return (ang < 0) ? ang + 2 * CV_PI : ang;
}

pair<double, vector<cv::Point2f>> get_bottom(vector<cv::Point2f> warp_corners, double orient,
                               double w, double h)
{
    logger->info("[{}:{}]", __func__, __LINE__);
    vector<cv::Point2f> a = to_iv(warp_corners);
    cv::Point2f k = get_intersect(a[3], cv::Point2f(a[3].x + cos(orient), a[3].y + sin(orient)), a[0],
                                  a[1]);

    if (k.x == numeric_limits<float>::infinity() || k.y == numeric_limits<float>::infinity())
    {
        logger->warn("[{}:{}]: lines are parallel", __func__, __LINE__);
        return make_pair(-1.0, vector<cv::Point2f>());
    }

    double l = norm(a[3] - k);

    if (l == 0)
    {
        logger->warn("[{}:{}]: l = 0", __func__, __LINE__);
        return make_pair(-1.0, vector<cv::Point2f>());
    }

    cv::Point2f c = ((l - w) * a[0] + w * a[3]) / l;
    //cv::Point2f c1 = ((l - w) * a[0] + w * a[3]) / l;
    //cv::Point2f c(c0.x, c0.y);

    cv::Point2f b = get_intersect(c, cv::Point2f(c.x + cos(orient), c.y + sin(orient)), a[0], a[1]);

    if (b.x == numeric_limits<float>::infinity()) // || b.y < a[0].y)
    {
        logger->warn("[{}:{}]: b = infinity", __func__, __LINE__);
        return make_pair(-1.0, vector<cv::Point2f>());
    }

    double ew = norm(c - b);
    double error_w = (ew < w) ? ew / w : w / ew;

    cv::Point2f d = get_intersect(c, cv::Point2f(c.x - sin(orient), c.y + cos(orient)), a[3], a[2]);

    if (d.x == numeric_limits<float>::infinity())
    {
        logger->warn("[{}:{}]: d = infinity", __func__, __LINE__);
        return make_pair(-1, vector<cv::Point2f>());
    }

    double el = norm(c - d);
    double error_l = (el < h) ? el / h : h / el;

    cv::Point2f center = (b + d)/2; //((b.x + d.x) / 2, (b.y + d.y) / 2);
    cv::Point2f f = 2 *center - c; //   (c.x + (center.x - c.x) * 2, c.y + (center.y - c.y) * 2);

    vector<cv::Point2f> w_points = to_iv(vector<cv::Point2f> {c, b, f, d});
    double error = error_w * error_l;

    return make_pair(error, w_points);
}

double get_motion_direction(const cv::Point2f direction, const std::vector<cv::Point2i> &box_2d,
                            int cls, const cv::Matx33f &persp_mat)
{
    int sum_x = 0;
    int sum_y = 0;

    for (const cv::Point2i &point : box_2d)
    {
        sum_x += point.x;
        sum_y += point.y;
    }

    cv::Point2f center(sum_x / 2, sum_y / 2);
    //direction /= norm(direction);
    // TODO without normalization
    cv::Point2f normalized_direction = direction / cv::norm(direction);
    //cv::normalize(direction, direction);

    try
    {
        cv::Point2f t1 = warp_perspective(center, persp_mat);
        cv::Point2f t2 = warp_perspective(center + normalized_direction, persp_mat);

        double v_x = t2.x - t1.x;
        double v_y = t1.y - t2.y;
        double angle = atan2(v_y, v_x);

        return angle;
    }
    catch (...)
    {
        logger->warn("Direction error");
        return numeric_limits<double>::infinity();
    }
}

std::pair<int, int> getObjSize(int cls)
{
    std::vector<std::pair<int, int>> sizes = {{258, 84}, {349, 165}, {370, 180}};
    return sizes[cls];
}

vector<cv::Point2f> get_upper_face_simple(vector<cv::Point2i> box_2d,
        vector<cv::Point2f> lower_face)
{
    logger->info("{}", __func__);
    vector<cv::Point2f> upper_face(4, cv::Point2f());

    double xmin = box_2d[0].x;
    double ymin = box_2d[0].y;
    double xmax = box_2d[1].x;
    double ymax = box_2d[1].y;

    double h_max = 0 - ymin + max(lower_face[0].y, lower_face[3].y);
    double h_min = 0 - ymin + min(lower_face[1].y, lower_face[2].y);

    upper_face[0] = lower_face[0] - cv::Point2f(0, h_min);
    upper_face[3] = lower_face[3] - cv::Point2f(0, h_min);

    upper_face[1] = lower_face[1] - cv::Point2f(0, h_min);
    upper_face[2] = lower_face[2] - cv::Point2f(0, h_min);

    //if (upper_face[3].y < ymin || upper_face[2].x < xmin || upper_face[1].x > xmax
        //|| upper_face[0].y > ymax)
    //{
        //return vector<cv::Point2f>();
    //}

    return upper_face;
}

std::vector<cv::Point2f> get_upper_face(const std::vector<cv::Point2i> &box_2d,
                                        const std::vector<cv::Point2f> &lower_face)
{
    logger->info("{}", __func__);
    return get_upper_face_simple(box_2d, lower_face);

    std::vector<cv::Point2f> upper_face(4);

    int xmin = box_2d[0].x;
    int ymin = box_2d[0].y;
    int xmax = box_2d[1].x;
    int ymax = box_2d[1].y;

    // E
    upper_face[2] = lower_face[2] - cv::Point2f(0, lower_face[2].y - ymin);

    // F
    try
    {
        cv::Point2f right_van, left_van, intersection;

        right_van = get_intersect(lower_face[1], lower_face[2], lower_face[0], lower_face[3]);
        intersection = get_intersect(upper_face[2], right_van, cv::Point2f(xmin, ymin), cv::Point2f(xmin,
                                     ymax));
        upper_face[1] = intersection;

        // G
        left_van = get_intersect(lower_face[2], lower_face[3], lower_face[0], lower_face[1]);
        intersection = get_intersect(upper_face[2], left_van, cv::Point2f(xmax, ymin), cv::Point2f(xmax,
                                     ymax));
        upper_face[3] = intersection;

        intersection = get_intersect(left_van, upper_face[1], right_van, upper_face[3]);
        upper_face[0] = intersection;
    }
    catch (...)
    {
        if (debug)
        {
            logger->warn("lines are parallel");
        }

        float k1 = cv::norm(lower_face[0] - lower_face[1]) / cv::norm(lower_face[2] - lower_face[3]);
        float k2 = cv::norm(lower_face[0] - lower_face[3]) / cv::norm(lower_face[2] - lower_face[1]);

        upper_face[1] = lower_face[1] - cv::Point2f(0, k1 * (lower_face[2].y - ymin));
        upper_face[3] = lower_face[3] - cv::Point2f(0, k2 * (lower_face[2].y - ymin));
        upper_face[0] = lower_face[0] - cv::Point2f(0, k1 * k2 * (lower_face[2].y - ymin));
    }

    // Check if points are inside the bounding box
    // for (const cv::Point2f& p : upper_face) {
    //     if (!check_in_bbox(box_2d, p)) {
    //         return get_simple_upper_face(box_2d, lower_face);
    //     }
    // }

    return upper_face;
}

Solution get_bottom_variants(const cv::Point2f &orig_mov_dir,
                             const std::vector<cv::Point2i> &box_2d,
                             const cv::Matx33f &mat, const cv::Matx33f &inv_mat,
                             const cv::Matx33f &inv_matrix_upper, int cls)
{
    logger->info("{}", __func__);

    // Get the motion angle
    double mov_angle = get_motion_direction(orig_mov_dir, box_2d, cls, mat);

    if (std::isinf(mov_angle))
    {
        logger->warn("Moving angle error");
        return Solution();
    }

    // Extract box corner coordinates
    int xmin = box_2d[0].x;
    int ymin = box_2d[0].y;
    int xmax = box_2d[1].x;
    int ymax = box_2d[1].y;

    // Define the 2D points of the box
    std::vector<cv::Point2f> ps = {cv::Point2f(xmin, ymax), cv::Point2f(xmin, ymin),
                                   cv::Point2f(xmax, ymin), cv::Point2f(xmax, ymax)
                                  };

    // Warp the box points
    std::vector<cv::Point2f> ps_bev = to_warp(ps, mat);

    // Get the object dimensions
    auto [dim_l, dim_w] = getObjSize(cls);

    if (std::isnan(dim_l) || std::isnan(dim_w))
    {
        if (debug)
        {
            //std::cout << "Object size error" << std::endl;
            logger->warn("Object size error");
        }
        return Solution();
    }

    // Compute the detected angle in degrees
    double detected_angle_deg = mov_angle * (180.0 / M_PI);

    // Calculate the bottom corners and error
    //double error;
    //std::vector<cv::Point2f> corners;
    auto [error, corners] = get_bottom(ps_bev, detected_angle_deg, dim_l, dim_w);

    if (corners.empty())
    {
        logger->warn("Corners not found");

        return Solution(cls, {}, {}, {}, {}, 0.0, 0.0, 0.0);
    }

    // Unwarp the bottom corners to the original frame
    std::vector<cv::Point2f> untop_corners = to_warp(corners, inv_mat);

    // Calculate height (h) relative to the original box
    float h = untop_corners[2].y - box_2d[1].y;

    // Calculate the upper face
    std::vector<cv::Point2f> upper_face = get_upper_face(box_2d, untop_corners);

    if (upper_face.empty())
    {
        return Solution(cls, {}, {}, {}, {}, 0.0, 0.0, 0.0);
    }

    // Calculate width error in pixels
    double w_error = error; // / pixels_in_mm (pixels_in_mm is undefined in your code)

    // Calculate height error
    cv::Point2f expected_edge = warp_perspective(corners[0], inv_matrix_upper);
    double expected_h = cv::norm(upper_face[0] - expected_edge);
    double computed_h = cv::norm(upper_face[0] - untop_corners[0]);

    double h_error = (computed_h < expected_h) ? (computed_h / expected_h) : (expected_h / computed_h);

    return Solution(cls, ps_bev, untop_corners, upper_face, corners, w_error, h_error,
                    detected_angle_deg);
}

cv::Matx33f get_mat_upper(int cls)
{
    double x_offset = 100;
    double y_offset = 6000;

    cv::Point2f src[4];

    if (cls == 2)
    {
        src[0] = cv::Point2f(215, 140);
        src[1] = cv::Point2f(90, 55);
        src[2] = cv::Point2f(294, 55);
        src[3] = cv::Point2f(625, 140);
    }
    else if (cls == 1)
    {
        src[0] = cv::Point2f(215, 185);
        src[1] = cv::Point2f(90, 80);
        src[2] = cv::Point2f(294, 80);
        src[3] = cv::Point2f(625, 185);
    }
    else if (cls == 0)
    {
        src[0] = cv::Point2f(215, 176);
        src[1] = cv::Point2f(90, 85);
        src[2] = cv::Point2f(294, 85);
        src[3] = cv::Point2f(625, 176);
    }

    double h = 7000;
    double w = 700;

    cv::Point2f dst[4];
    dst[0] = cv::Point2f(0 + x_offset, 0 + h + y_offset);
    dst[1] = cv::Point2f(0 + x_offset, 0 + y_offset);
    dst[2] = cv::Point2f(w + x_offset, 0 + y_offset);
    dst[3] = cv::Point2f(w + x_offset, h + y_offset);

    return cv::getPerspectiveTransform(dst, src);
}

pair<cv::Matx33f, cv::Matx33f> get_mat()
{
    double x_offset = 100;
    double y_offset = 6000;

    cv::Point2f src[4];
    src[0] = cv::Point2f(215, 265);
    src[1] = cv::Point2f(90, 121);
    src[2] = cv::Point2f(294, 120);
    src[3] = cv::Point2f(625, 265);

    double k = VEHICLE_L / VEHICLE_W;
    double w = norm(src[0] - src[3]);
    double h = k * w;

    double vehicle_w_bev = norm(cv::Point2f(169, 13513) - cv::Point2f(325, 13453));
    pixels_in_mm = vehicle_w_bev / VEHICLE_W;
    mm_in_pix = VEHICLE_W / vehicle_w_bev;

    h = 7000;
    w = 700;

    cv::Point2f dst[4];
    dst[0] = cv::Point2f(0 + x_offset, 0 + h + y_offset);
    dst[1] = cv::Point2f(0 + x_offset, 0 + y_offset);
    dst[2] = cv::Point2f(w + x_offset, 0 + y_offset);
    dst[3] = cv::Point2f(w + x_offset, h + y_offset);

    cv::Matx33f mat = cv::getPerspectiveTransform(src, dst);
    cv::Matx33f inv_mat = cv::getPerspectiveTransform(dst, src);

    return make_pair(mat, inv_mat);
}

cv::Mat transform(cv::Mat img, cv::Matx33f mat)
{
    cv::Size dim(IMAGE_W, IMAGE_H);
    cv::Mat result;
    cv::warpPerspective(img, result, mat, dim, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return result;
}

pair<double, double> get_obj_size(int cls)
{
    vector<pair<double, double>> sizes = {{258, 84}, {349, 165}, {370, 180}};
    return sizes[cls];
}

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

const bool debug = false;

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

cv::Point2f get_intersect(cv::Point2f a1, cv::Point2f a2, cv::Point2f b1, cv::Point2f b2) {
    cv::Matx33f A(a1.x - a2.x, b2.x - b1.x, a1.x - b1.x,
                   a1.y - a2.y, b2.y - b1.y, a1.y - b1.y,
                   0, 0, 1);

    cv::Vec3f B(b1.x - a2.x, b1.y - a2.y, 0);

    cv::Vec3f X = A.inv() * B;

    return cv::Point2f(X[0], X[1]);
}

cv::Point2f warp_perspective(cv::Point2f p, cv::Matx33f matrix) {
    float px = (matrix(0, 0) * p.x + matrix(0, 1) * p.y + matrix(0, 2)) /
               (matrix(2, 0) * p.x + matrix(2, 1) * p.y + matrix(2, 2));

    float py = (matrix(1, 0) * p.x + matrix(1, 1) * p.y + matrix(1, 2)) /
               (matrix(2, 0) * p.x + matrix(2, 1) * p.y + matrix(2, 2));

    return cv::Point2f(px, py);
}

vector<cv::Point2f> warp(vector<cv::Point2f> a, int i, cv::Matx33f matrix) {
    vector<cv::Point2f> result;
    result.push_back(warp_perspective(a[i], matrix));
    return result;
}

vector<cv::Point2f> to_warp(vector<cv::Point2f> a, cv::Matx33f matrix) {
    vector<cv::Point2f> result;
    for (int i = 0; i < a.size(); i++) {
        vector<cv::Point2f> warped = warp(a, i, matrix);
        result.insert(result.end(), warped.begin(), warped.end());
    }
    return result;
}

cv::Point2f iv(cv::Point2f a) {
    return cv::Point2f(a.x, -a.y);
}

vector<cv::Point2f> to_iv(vector<cv::Point2f> array) {
    vector<cv::Point2f> converted;
    for (int i = 0; i < array.size(); i++) {
        converted.push_back(iv(array[i]));
    }
    return converted;
}

double get_angle(cv::Point2f a, cv::Point2f b, cv::Point2f c) {
    double ang = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x);
    return (ang < 0) ? ang + 2 * CV_PI : ang;
}

pair<double, vector<cv::Point2f>> get_bottom(vector<cv::Point2f> warp_corners, double orient, double w, double h) {
    vector<cv::Point2f> a = to_iv(warp_corners);
    cv::Point2f k = get_intersect(a[3], cv::Point2f(a[3].x + cos(orient), a[3].y + sin(orient)), a[0], a[1]);

    if (k.x == numeric_limits<float>::infinity() || k.y == numeric_limits<float>::infinity()) {
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    double l = norm(a[3] - k);

    if (l == 0) {
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    cv::Point2f c0 = ((l - w) * a[0] + w * a[3]) / l;
    cv::Point2f c1 = ((l - w) * a[0] + w * a[3]) / l;
    cv::Point2f c(c0.x, c0.y);

    cv::Point2f b = get_intersect(c, cv::Point2f(c.x + cos(orient), c.y + sin(orient)), a[0], a[1]);

    if (b.x == numeric_limits<float>::infinity() || b.y < a[0].y) {
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    double ew = norm(c - b);
    double error_w = (ew < w) ? ew / w : w / ew;

    cv::Point2f d = get_intersect(c, cv::Point2f(c.x - sin(orient), c.y + cos(orient)), a[3], a[2]);

    if (d.x == numeric_limits<float>::infinity()) {
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    double el = norm(c - d);
    double error_l = (el < h) ? el / h : h / el;

    cv::Point2f center((b.x + d.x) / 2, (b.y + d.y) / 2);
    cv::Point2f f(c.x + (center.x - c.x) * 2, c.y + (center.y - c.y) * 2);

    vector<cv::Point2f> w_points = to_iv(vector<cv::Point2f>{c, b, f, d});
    double error = error_w * error_l;

    return make_pair(error, w_points);
}

pair<double, vector<cv::Point2f>> get_frame(vector<cv::Point2f> warp_corners, double orient, double l_o, double w_o,
                                           cv::Mat bev_in, vector<int> color) {
    if (orient < 0 && orient > -CV_PI / 2) {
        w_o = l_o;
    } else if (orient < 0 && orient > -CV_PI) {
        w_o = l_o;
    } else {
        cout << "Moving angle detection error" << endl;
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    vector<cv::Point2f> a = to_iv(warp_corners);
    cv::Point2f k = get_intersect(a[3], cv::Point2f(a[3].x + cos(orient), a[3].y + sin(orient)), a[0], a[1]);

    if (k.x == numeric_limits<float>::infinity() || k.y == numeric_limits<float>::infinity()) {
        orient = orient + CV_PI / 2;
        k = get_intersect(a[3], cv::Point2f(a[3].x + cos(orient), a[3].y + sin(orient)), a[0], a[1]);
    }

    if (k.x == numeric_limits<float>::infinity() || k.y == numeric_limits<float>::infinity()) {
        cout << "No intersection for k" << endl;
        cv::Point2f c = a[0];
        cv::Point2f d = a[3];
        double v_x = a[1].x - a[0].x;
        double v_y = a[1].y - a[0].y;
        double angle = atan2(v_y, v_x);
        cv::Point2f b(a[0].x + h * cos(angle), a[0].y + h * sin(angle));
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    if (k.y + numeric_limits<float>::epsilon() < a[0].y) {
        cout << "k out of bbbox" << endl;
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    cv::Point2f c0 = ((w * h - w_o * w) * a[0].x + w_o * w * k.x) / (w * h - w_o * w);
    cv::Point2f c1 = ((w * h - w_o * w) * a[0].y + w_o * w * k.y) / (w * h - w_o * w);
    cv::Point2f c(c0.x, c0.y);

    if (color.size() != 0 && c.x > 2 * l_o / 3) {
        color[0] = 0;
    }

    cv::Point2f b = get_intersect(c, cv::Point2f(c.x + cos(orient), c.y + sin(orient)), a[0], a[1]);

    if (b.x == numeric_limits<float>::infinity() || b.y < a[0].y) {
        cout << "No intersection for b" << endl;
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    double ew = norm(c - b);
    double error_w = ew / w;

    cv::Point2f d = get_intersect(c, cv::Point2f(c.x - sin(orient), c.y + cos(orient)), a[3], a[2]);

    if (d.x == numeric_limits<float>::infinity()) {
        return make_pair(numeric_limits<double>::infinity(), vector<cv::Point2f>());
    }

    double el = norm(c - d);
    double error_l = el / h;

    cv::Point2f center((b.x + d.x) / 2, (b.y + d.y) / 2);
    cv::Point2f f(c.x + (center.x - c.x) * 2, c.y + (center.y - c.y) * 2);

    vector<cv::Point2f> w_points = to_iv(vector<cv::Point2f>{c, b, f, d});
    double error = error_w * error_l;

    return make_pair(error, w_points);
}

pair<vector<cv::Point2f>, vector<cv::Point2f>> get_bottom_variants(cv::Point2f orig_mov_dir, vector<float> box_2d,
                                                                cv::Matx33f mat, cv::Matx33f inv_mat,
                                                                cv::Matx33f inv_matrix_upper, int cls, bool debug = false) {
    if (debug) {
        cout << "orig_mov_dir: " << orig_mov_dir << endl;
    }

    double mov_angle = get_motion_direction(orig_mov_dir, box_2d, cls, mat);

    if (mov_angle == numeric_limits<double>::infinity()) {
        if (debug) {
            cout << "Moving angle error" << endl;
        }
        return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
    }

    double xmin = box_2d[0];
    double ymin = box_2d[1];
    double xmax = box_2d[2];
    double ymax = box_2d[3];

    vector<cv::Point2f> ps;
    ps.push_back(cv::Point2f(xmin, ymax));
    ps.push_back(cv::Point2f(xmin, ymin));
    ps.push_back(cv::Point2f(xmax, ymin));
    ps.push_back(cv::Point2f(xmax, ymax));

    vector<cv::Point2f> ps_bev = to_warp(ps, mat);

    pair<double, vector<cv::Point2f>> obj_size = get_obj_size(cls);

    if (obj_size.first == numeric_limits<double>::infinity() || obj_size.second == numeric_limits<double>::infinity()) {
        if (debug) {
            cout << "Object size error" << endl;
        }
        return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
    }

    double detected_angle_deg = mov_angle * 180.0 / CV_PI;

    pair<double, vector<cv::Point2f>> error_corners = get_bottom(ps_bev, mov_angle, obj_size.first, obj_size.second);

    if (error_corners.first == numeric_limits<double>::infinity() || error_corners.second.size() == 0) {
        if (debug) {
            cout << "Corners not found" << endl;
        }
        return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
    }

    vector<cv::Point2f> untop_corners;
    for (int i = 0; i < error_corners.second.size(); i++) {
        vector<cv::Point2f> unwarped = warp(error_corners.second, i, inv_mat);
        untop_corners.insert(untop_corners.end(), unwarped.begin(), unwarped.end());
    }

    double h = untop_corners[2].y - box_2d[1];

    if (error_corners.second[2].x > xmax) {
        return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
    }

    vector<cv::Point2f> lower_face = untop_corners;
    vector<cv::Point2f> upper_face = get_upper_face(box_2d, lower_face);

    if (upper_face.empty() || upper_face[2].x > xmax) {
        return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
    }

    double w_error = error_corners.first;

    cv::Point2f expected_edge = warp_perspective(error_corners.second[0], inv_matrix_upper);
    double expected_h = norm(lower_face[0] - expected_edge);
    double computed_h = norm(lower_face[0] - upper_face[0]);

    double h_error = 0.0;

    if (computed_h < expected_h) {
        h_error = computed_h / expected_h;
    } else {
        h_error = expected_h / computed_h;
    }

    return make_pair(ps_bev, lower_face);
}

cv::Matx33f get_mat_upper(int cls) {
    double x_offset = 100;
    double y_offset = 6000;

    cv::Point2f src[4];

    if (cls == 2) {
        src[0] = cv::Point2f(215, 140);
        src[1] = cv::Point2f(90, 55);
        src[2] = cv::Point2f(294, 55);
        src[3] = cv::Point2f(625, 140);
    } else if (cls == 1) {
        src[0] = cv::Point2f(215, 185);
        src[1] = cv::Point2f(90, 80);
        src[2] = cv::Point2f(294, 80);
        src[3] = cv::Point2f(625, 185);
    } else if (cls == 0) {
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

pair<cv::Matx33f, cv::Matx33f> get_mat() {
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

cv::Mat transform(cv::Mat img, cv::Matx33f mat) {
    cv::Size dim(IMAGE_W, IMAGE_H);
    cv::Mat result;
    cv::warpPerspective(img, result, mat, dim, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return result;
}

double get_motion_direction(cv::Point2f direction, vector<float> box_2d, int cls, cv::Matx33f persp_mat) {
    double x1 = box_2d[0];
    double y1 = box_2d[1];
    double x2 = box_2d[2];
    double y2 = box_2d[3];

    cv::Point2f center((x1 + x2) / 2, (y1 + y2) / 2);
    direction /= norm(direction);

    try {
        cv::Point2f t1 = warp_perspective(center, persp_mat);
        cv::Point2f t2 = warp_perspective(center + direction, persp_mat);

        double v_x = t2.x - t1.x;
        double v_y = t1.y - t2.y;
        double angle = atan2(v_y, v_x);

        return angle;
    } catch (...) {
        cout << "Direction error" << endl;
        return numeric_limits<double>::infinity();
    }
}

pair<double, double> get_obj_size(int cls) {
    vector<pair<double, double>> sizes = {{258, 84}, {349, 165}, {370, 180}};
    return sizes[cls];
}

vector<cv::Point2f> get_upper_face(vector<float> box_2d, vector<cv::Point2f> lower_face) {
    vector<cv::Point2f> upper_face(4, cv::Point2f());

    double xmin = box_2d[0];
    double ymin = box_2d[1];
    double xmax = box_2d[2];
    double ymax = box_2d[3];

    double h_max = 0 - ymin + max(lower_face[0].y, lower_face[3].y);
    double h_min = 0 - ymin + min(lower_face[1].y, lower_face[2].y);

    upper_face[0] = lower_face[0] - cv::Point2f(0, h_min);
    upper_face[3] = lower_face[3] - cv::Point2f(0, h_min);

    upper_face[1] = lower_face[1] - cv::Point2f(0, h_min);
    upper_face[2] = lower_face[2] - cv::Point2f(0, h_min);

    if (upper_face[3].y < ymin || upper_face[2].x < xmin || upper_face[1].x > xmax || upper_face[0].y > ymax) {
        return vector<cv::Point2f>();
    }

    return upper_face;
}

vector<cv::Point2f> get_upper_face_variant(vector<float> box_2d, vector<cv::Point2f> lower_face) {
    vector<cv::Point2f> upper_face(4, cv::Point2f());

    double xmin = box_2d[0];
    double ymin = box_2d[1];
    double xmax = box_2d[2];
    double ymax = box_2d[3];

    double h_max = 0 - ymin + max(lower_face[0].y, lower_face[3].y);
    double h_min = 0 - ymin + min(lower_face[1].y, lower_face[2].y);

    upper_face[0] = lower_face[0] - cv::Point2f(0, h_min);
    upper_face[3] = lower_face[3] - cv::Point2f(0, h_min);

    upper_face[1] = lower_face[1] - cv::Point2f(0, h_min);
    upper_face[2] = lower_face[2] - cv::Point2f(0, h_min);

    if (upper_face[3].y < ymin || upper_face[2].x < xmin || upper_face[1].x > xmax || upper_face[0].y > ymax) {
        return vector<cv::Point2f>();
    }

    return upper_face;
}

vector<cv::Point2f> draw_vehicle_bev(vector<cv::Point2f> corners, vector<float> box_2d, cv::Mat frame_bev,
                                     cv::Mat *box_2d_img = nullptr) {
    vector<cv::Point2f> lower_face;
    vector<cv::Point2f> lower_face_variant;

    cv::Mat box_2d_img;
    if (box_2d_img == nullptr) {
        box_2d_img = frame_bev.clone();
    }

    int min_idx = -1;
    double min_error = numeric_limits<double>::infinity();

    int min_idx_variant = -1;
    double min_error_variant = numeric_limits<double>::infinity();

    cv::Matx33f persp_mat;
    cv::Matx33f persp_mat_upper;

    tie(persp_mat, persp_mat_upper) = get_mat();
    cv::Matx33f inv_persp_mat = persp_mat.inv();
    cv::Matx33f inv_persp_mat_upper = persp_mat_upper.inv();

    for (int i = 0; i < 2; i++) {
        pair<vector<cv::Point2f>, vector<cv::Point2f>> box_data = get_bottom_variants(corners[1], box_2d, persp_mat,
                                                                                      inv_persp_mat,
                                                                                      inv_persp_mat_upper, i);

        vector<cv::Point2f> ps_bev = box_data.first;
        vector<cv::Point2f> ps_upper = box_data.second;

        if (ps_upper.size() == 0) {
            continue;
        }

        pair<double, double> obj_size = get_obj_size(i);

        if (obj_size.first == numeric_limits<double>::infinity() || obj_size.second == numeric_limits<double>::infinity()) {
            continue;
        }

        double w_o = obj_size.first;
        double h_o = obj_size.second;

        double x_offset = 100;
        double y_offset = 6000;

        vector<cv::Point2f> ps_img;
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[0].x * mm_in_pix, y_offset + ps_bev[0].y * mm_in_pix));
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[1].x * mm_in_pix, y_offset + ps_bev[1].y * mm_in_pix));
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[2].x * mm_in_pix, y_offset + ps_bev[2].y * mm_in_pix));
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[3].x * mm_in_pix, y_offset + ps_bev[3].y * mm_in_pix));

        if (debug) {
            for (int i = 0; i < 4; i++) {
                circle(frame_bev, ps_bev[i], 15, Scalar(0, 0, 255), -1);
            }
            line(frame_bev, ps_bev[0], ps_bev[1], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_bev[1], ps_bev[2], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_bev[2], ps_bev[3], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_bev[3], ps_bev[0], Scalar(0, 255, 0), 5);

            for (int i = 0; i < 4; i++) {
                circle(frame_bev, ps_upper[i], 15, Scalar(0, 255, 0), -1);
            }
            line(frame_bev, ps_upper[0], ps_upper[1], Scalar(0, 0, 255), 5);
            line(frame_bev, ps_upper[1], ps_upper[2], Scalar(0, 0, 255), 5);
            line(frame_bev, ps_upper[2], ps_upper[3], Scalar(0, 0, 255), 5);
            line(frame_bev, ps_upper[3], ps_upper[0], Scalar(0, 0, 255), 5);

            line(frame_bev, ps_bev[0], ps_upper[0], Scalar(255, 0, 0), 5);
            line(frame_bev, ps_bev[1], ps_upper[1], Scalar(255, 0, 0), 5);
            line(frame_bev, ps_bev[2], ps_upper[2], Scalar(255, 0, 0), 5);
            line(frame_bev, ps_bev[3], ps_upper[3], Scalar(255, 0, 0), 5);
        }

        double xmin = box_2d[0];
        double ymin = box_2d[1];
        double xmax = box_2d[2];
        double ymax = box_2d[3];

        double h_max = 0 - ymin + max(ps_upper[0].y, ps_upper[3].y);
        double h_min = 0 - ymin + min(ps_upper[1].y, ps_upper[2].y);

        vector<cv::Point2f> upper_face = ps_upper;
        vector<cv::Point2f> lower_face = ps_bev;

        upper_face[0] = ps_upper[0] - cv::Point2f(0, h_min);
        upper_face[3] = ps_upper[3] - cv::Point2f(0, h_min);

        upper_face[1] = ps_upper[1] - cv::Point2f(0, h_min);
        upper_face[2] = ps_upper[2] - cv::Point2f(0, h_min);

        if (debug) {
            for (int i = 0; i < 4; i++) {
                circle(frame_bev, upper_face[i], 15, Scalar(255, 0, 0), -1);
            }
            line(frame_bev, upper_face[0], upper_face[1], Scalar(255, 0, 0), 5);
            line(frame_bev, upper_face[1], upper_face[2], Scalar(255, 0, 0), 5);
            line(frame_bev, upper_face[2], upper_face[3], Scalar(255, 0, 0), 5);
            line(frame_bev, upper_face[3], upper_face[0], Scalar(255, 0, 0), 5);
        }

        vector<cv::Point2f> untop_corners;
        vector<cv::Point2f> upper_face_variant;
        upper_face_variant = get_upper_face_variant(box_2d, lower_face);

        if (upper_face_variant.size() != 0) {
            double h_max = 0 - ymin + max(upper_face_variant[0].y, upper_face_variant[3].y);
            double h_min = 0 - ymin + min(upper_face_variant[1].y, upper_face_variant[2].y);

            upper_face_variant[0] = upper_face_variant[0] - cv::Point2f(0, h_min);
            upper_face_variant[3] = upper_face_variant[3] - cv::Point2f(0, h_min);

            upper_face_variant[1] = upper_face_variant[1] - cv::Point2f(0, h_min);
            upper_face_variant[2] = upper_face_variant[2] - cv::Point2f(0, h_min);

            upper_face = upper_face_variant;

            if (debug) {
                for (int i = 0; i < 4; i++) {
                    circle(frame_bev, upper_face_variant[i], 15, Scalar(0, 0, 255), -1);
                }
                line(frame_bev, upper_face_variant[0], upper_face_variant[1], Scalar(0, 0, 255), 5);
                line(frame_bev, upper_face_variant[1], upper_face_variant[2], Scalar(0, 0, 255), 5);
                line(frame_bev, upper_face_variant[2], upper_face_variant[3], Scalar(0, 0, 255), 5);
                line(frame_bev, upper_face_variant[3], upper_face_variant[0], Scalar(0, 0, 255), 5);
            }
        }

        for (int i = 0; i < upper_face.size(); i++) {
            vector<cv::Point2f> unwarped = warp(upper_face, i, inv_persp_mat);
            untop_corners.insert(untop_corners.end(), unwarped.begin(), unwarped.end());
        }

        vector<cv::Point2f> upper_face_unwarped;
        for (int i = 0; i < upper_face.size(); i++) {
            vector<cv::Point2f> unwarped = warp(upper_face, i, inv_persp_mat_upper);
            upper_face_unwarped.insert(upper_face_unwarped.end(), unwarped.begin(), unwarped.end());
        }

        double error = box_error(h_o, w_o, obj_size.first, obj_size.second, upper_face_unwarped, upper_face_unwarped[2],
                                 upper_face_unwarped[3], ps_img, untop_corners, obj_size.first, obj_size.second);

        if (debug) {
            cout << "Error: " << error << endl;
        }

        if (error < min_error) {
            min_error = error;
            min_idx = i;
            lower_face = ps_bev;
        }

        if (debug) {
            line(box_2d_img, cv::Point2f(xmin, ymin), cv::Point2f(xmin, ymax), Scalar(0, 0, 255), 5);
            line(box_2d_img, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymin), Scalar(0, 0, 255), 5);
            line(box_2d_img, cv::Point2f(xmin, ymax), cv::Point2f(xmax, ymax), Scalar(0, 0, 255), 5);
            line(box_2d_img, cv::Point2f(xmax, ymin), cv::Point2f(xmax, ymax), Scalar(0, 0, 255), 5);

            line(box_2d_img, ps_img[0], ps_img[1], Scalar(0, 255, 0), 5);
            line(box_2d_img, ps_img[1], ps_img[2], Scalar(0, 255, 0), 5);
            line(box_2d_img, ps_img[2], ps_img[3], Scalar(0, 255, 0), 5);
            line(box_2d_img, ps_img[3], ps_img[0], Scalar(0, 255, 0), 5);
        }

        if (box_2d_img != nullptr) {
            cv::imshow("box_2d_img", box_2d_img);
        }
    }

    if (min_idx == -1) {
        return vector<cv::Point2f>();
    }

    if (debug) {
        cout << "Lower face selected: " << min_idx << endl;
    }

    if (lower_face.size() != 0) {
        vector<cv::Point2f> untop_corners;
        for (int i = 0; i < lower_face.size(); i++) {
            vector<cv::Point2f> unwarped = warp(lower_face, i, inv_persp_mat);
            untop_corners.insert(untop_corners.end(), unwarped.begin(), unwarped.end());
        }

        double w_o = box_2d[2] - box_2d[0];
        double h_o = box_2d[3] - box_2d[1];

        vector<cv::Point2f> ps_bev = lower_face;
        double x_offset = 100;
        double y_offset = 6000;

        vector<cv::Point2f> ps_img;
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[0].x * mm_in_pix, y_offset + ps_bev[0].y * mm_in_pix));
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[1].x * mm_in_pix, y_offset + ps_bev[1].y * mm_in_pix));
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[2].x * mm_in_pix, y_offset + ps_bev[2].y * mm_in_pix));
        ps_img.push_back(cv::Point2f(x_offset + ps_bev[3].x * mm_in_pix, y_offset + ps_bev[3].y * mm_in_pix));

        double h_max = 0 - box_2d[1] + max(lower_face[0].y, lower_face[3].y);
        double h_min = 0 - box_2d[1] + min(lower_face[1].y, lower_face[2].y);

        lower_face[0] = lower_face[0] - cv::Point2f(0, h_min);
        lower_face[3] = lower_face[3] - cv::Point2f(0, h_min);

        lower_face[1] = lower_face[1] - cv::Point2f(0, h_min);
        lower_face[2] = lower_face[2] - cv::Point2f(0, h_min);

        if (box_2d_img != nullptr) {
            line(box_2d_img, cv::Point2f(box_2d[0], box_2d[1]), cv::Point2f(box_2d[0], box_2d[3]), Scalar(0, 255, 0), 5);
            line(box_2d_img, cv::Point2f(box_2d[0], box_2d[1]), cv::Point2f(box_2d[2], box_2d[1]), Scalar(0, 255, 0), 5);
            line(box_2d_img, cv::Point2f(box_2d[0], box_2d[3]), cv::Point2f(box_2d[2], box_2d[3]), Scalar(0, 255, 0), 5);
            line(box_2d_img, cv::Point2f(box_2d[2], box_2d[1]), cv::Point2f(box_2d[2], box_2d[3]), Scalar(0, 255, 0), 5);

            line(box_2d_img, ps_img[0], ps_img[1], Scalar(0, 0, 255), 5);
            line(box_2d_img, ps_img[1], ps_img[2], Scalar(0, 0, 255), 5);
            line(box_2d_img, ps_img[2], ps_img[3], Scalar(0, 0, 255), 5);
            line(box_2d_img, ps_img[3], ps_img[0], Scalar(0, 0, 255), 5);
        }

        if (box_2d_img != nullptr) {
            cv::imshow("box_2d_img", box_2d_img);
            cv::waitKey(0);
        }

        return ps_bev;
    } else {
        return vector<cv::Point2f>();
    }
}

vector<cv::Point2f> warp(vector<cv::Point2f> corners, int i, cv::Matx33f inv_persp_mat) {
    vector<cv::Point2f> untop_corners;

    vector<cv::Point2f> dst(1, corners[i]);
    cv::perspectiveTransform(dst, untop_corners, inv_persp_mat);

    return untop_corners;
}

pair<vector<cv::Point2f>, vector<cv::Point2f>> get_bottom_variants(cv::Point2f start, vector<float> box_2d,
                                                                   cv::Matx33f persp_mat, cv::Matx33f inv_persp_mat,
                                                                   cv::Matx33f inv_persp_mat_upper, int idx) {
    pair<vector<cv::Point2f>, vector<cv::Point2f>> box_data;
    if (idx == 0) {
        box_data = get_bottom_corners_left(start, box_2d, persp_mat, inv_persp_mat, inv_persp_mat_upper);
    } else {
        box_data = get_bottom_corners_right(start, box_2d, persp_mat, inv_persp_mat, inv_persp_mat_upper);
    }
    return box_data;
}

pair<vector<cv::Point2f>, vector<cv::Point2f>> get_bottom_corners_left(cv::Point2f start, vector<float> box_2d,
                                                                       cv::Matx33f persp_mat,
                                                                       cv::Matx33f inv_persp_mat,
                                                                       cv::Matx33f inv_persp_mat_upper) {
    vector<cv::Point2f> lower_face = get_lower_face(box_2d, persp_mat, inv_persp_mat, inv_persp_mat_upper);

    if (lower_face.empty() || lower_face[2].x < start.x) {
        return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
    }

    vector<cv::Point2f> ps_bev;
    vector<cv::Point2f> upper_face;

    int cls = get_class(box_2d, true);

    if (cls == 2) {
        upper_face = get_upper_face(box_2d, lower_face);

        if (upper_face.empty() || upper_face[2].x > start.x) {
            return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
        }

        ps_bev.push_back(start);
        ps_bev.push_back(cv::Point2f(start.x, lower_face[1].y));
        ps_bev.push_back(lower_face[2]);
        ps_bev.push_back(lower_face[3]);
    } else if (cls == 1) {
        upper_face = get_upper_face(box_2d, lower_face);

        if (upper_face.empty() || upper_face[2].x > start.x) {
            return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
        }

        ps_bev.push_back(start);
        ps_bev.push_back(cv::Point2f(start.x, start.y - (lower_face[0].y - lower_face[1].y)));
        ps_bev.push_back(lower_face[2]);
        ps_bev.push_back(lower_face[3]);
    } else if (cls == 0) {
        upper_face = get_upper_face(box_2d, lower_face);

        if (upper_face.empty() || upper_face[2].x > start.x) {
            return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
        }

        ps_bev.push_back(start);
        ps_bev.push_back(cv::Point2f(start.x, start.y - (lower_face[0].y - lower_face[1].y)));
        ps_bev.push_back(lower_face[2]);
        ps_bev.push_back(lower_face[3]);
    }

    return make_pair(ps_bev, lower_face);
}

pair<vector<cv::Point2f>, vector<cv::Point2f>> get_bottom_corners_right(cv::Point2f start, vector<float> box_2d,
                                                                        cv::Matx33f persp_mat,
                                                                        cv::Matx33f inv_persp_mat,
                                                                        cv::Matx33f inv_persp_mat_upper) {
    vector<cv::Point2f> lower_face = get_lower_face(box_2d, persp_mat, inv_persp_mat, inv_persp_mat_upper);

    if (lower_face.empty() || lower_face[1].x > start.x) {
        return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
    }

    vector<cv::Point2f> ps_bev;
    vector<cv::Point2f> upper_face;

    int cls = get_class(box_2d, true);

    if (cls == 2) {
        upper_face = get_upper_face(box_2d, lower_face);

        if (upper_face.empty() || upper_face[1].x < start.x) {
            return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
        }

        ps_bev.push_back(start);
        ps_bev.push_back(cv::Point2f(start.x, lower_face[0].y));
        ps_bev.push_back(lower_face[1]);
        ps_bev.push_back(lower_face[2]);
    } else if (cls == 1) {
        upper_face = get_upper_face(box_2d, lower_face);

        if (upper_face.empty() || upper_face[1].x < start.x) {
            return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
        }

        ps_bev.push_back(start);
        ps_bev.push_back(cv::Point2f(start.x, start.y - (lower_face[3].y - lower_face[2].y)));
        ps_bev.push_back(lower_face[1]);
        ps_bev.push_back(lower_face[2]);
    } else if (cls == 0) {
        upper_face = get_upper_face(box_2d, lower_face);

        if (upper_face.empty() || upper_face[1].x < start.x) {
            return make_pair(vector<cv::Point2f>(), vector<cv::Point2f>());
        }

        ps_bev.push_back(start);
        ps_bev.push_back(cv::Point2f(start.x, start.y - (lower_face[3].y - lower_face[2].y)));
        ps_bev.push_back(lower_face[1]);
        ps_bev.push_back(lower_face[2]);
    }

    return make_pair(ps_bev, lower_face);
}

vector<cv::Point2f> get_lower_face(vector<float> box_2d, cv::Matx33f persp_mat, cv::Matx33f inv_persp_mat,
                                    cv::Matx33f inv_persp_mat_upper) {
    vector<cv::Point2f> corners_2d;

    vector<cv::Point2f> ps_img;

    corners_2d.push_back(cv::Point2f(box_2d[0], box_2d[1]));
    corners_2d.push_back(cv::Point2f(box_2d[2], box_2d[1]));
    corners_2d.push_back(cv::Point2f(box_2d[2], box_2d[3]));
    corners_2d.push_back(cv::Point2f(box_2d[0], box_2d[3]));

    vector<cv::Point2f> ps_bev;
    vector<cv::Point2f> ps_upper;
    vector<cv::Point2f> ps_lower;

    vector<cv::Point2f> ps_bev_transformed;

    for (int i = 0; i < 4; i++) {
        vector<cv::Point2f> dst(1, corners_2d[i]);
        perspectiveTransform(dst, ps_img, persp_mat);
        ps_bev.push_back(ps_img[0]);

        vector<cv::Point2f> untop_corners;
        vector<cv::Point2f> untop_corners_variant;

        untop_corners = warp(corners_2d, i, inv_persp_mat);
        untop_corners_variant = warp(corners_2d, i, inv_persp_mat_upper);

        if (untop_corners.empty() || untop_corners_variant.empty()) {
            return vector<cv::Point2f>();
        }

        if (ps_img[0].y >= 6000) {
            return vector<cv::Point2f>();
        }

        if (ps_img[0].y >= 5500) {
            continue;
        }

        if (i == 0) {
            if (abs(ps_img[0].x - 150) > 30) {
                return vector<cv::Point2f>();
            }
        } else if (i == 1) {
            if (abs(ps_img[0].x - 300) > 30) {
                return vector<cv::Point2f>();
            }
        } else if (i == 2) {
            if (abs(ps_img[0].x - 150) > 30) {
                return vector<cv::Point2f>();
            }
        } else if (i == 3) {
            if (abs(ps_img[0].x - 300) > 30) {
                return vector<cv::Point2f>();
            }
        }

        ps_bev_transformed.push_back(ps_img[0]);
        ps_upper.push_back(untop_corners[0]);
        ps_lower.push_back(untop_corners_variant[0]);
    }

    int cls = get_class(box_2d, true);

    if (cls == 2) {
        if (debug) {
            circle(frame_bev, ps_upper[0], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_upper[1], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_lower[2], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_lower[3], 15, Scalar(0, 255, 0), -1);
        }
    } else if (cls == 1) {
        if (debug) {
            circle(frame_bev, ps_lower[0], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_upper[1], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_lower[2], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_lower[3], 15, Scalar(0, 255, 0), -1);
        }
    } else if (cls == 0) {
        if (debug) {
            circle(frame_bev, ps_lower[0], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_upper[1], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_lower[2], 15, Scalar(0, 255, 0), -1);
            circle(frame_bev, ps_lower[3], 15, Scalar(0, 255, 0), -1);
        }
    }

    if (cls == 2) {
        if (debug) {
            line(frame_bev, ps_upper[0], ps_upper[1], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_upper[1], ps_lower[2], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_lower[2], ps_lower[3], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_lower[3], ps_upper[0], Scalar(0, 255, 0), 5);
        }

        return ps_bev;
    } else if (cls == 1) {
        if (debug) {
            line(frame_bev, ps_lower[0], ps_upper[1], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_upper[1], ps_lower[2], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_lower[2], ps_lower[3], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_lower[3], ps_lower[0], Scalar(0, 255, 0), 5);
        }

        return ps_bev;
    } else if (cls == 0) {
        if (debug) {
            line(frame_bev, ps_lower[0], ps_upper[1], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_upper[1], ps_lower[2], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_lower[2], ps_lower[3], Scalar(0, 255, 0), 5);
            line(frame_bev, ps_lower[3], ps_lower[0], Scalar(0, 255, 0), 5);
        }

        return ps_bev;
    }
}

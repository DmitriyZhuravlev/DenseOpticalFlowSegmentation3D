#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

void addTextToImage(
    cv::Mat& image_rgb,
    const std::string& label,
    const std::tuple<int, int>& top_left_xy = std::make_tuple(0, 0),
    double font_scale = 1,
    int font_thickness = 1,
    int font_face = cv::FONT_HERSHEY_SIMPLEX,
    const cv::Scalar& font_color_rgb = cv::Scalar(0, 0, 255),
    const cv::Scalar& bg_color_rgb = cv::Scalar(0, 0, 0),
    const cv::Scalar& outline_color_rgb = cv::Scalar(0, 0, 0),
    double line_spacing = 1.0
) {
    const int OUTLINE_FONT_THICKNESS = 3 * font_thickness;

    int im_h = image_rgb.rows;
    int im_w = image_rgb.cols;

    std::vector<std::string> lines;
    size_t pos = 0;
    while ((pos = label.find('\n')) != std::string::npos) {
        lines.push_back(label.substr(0, pos));
        label.erase(0, pos + 1);
    }
    if (!label.empty()) {
        lines.push_back(label);
    }

    for (const std::string& line : lines) {
        int x, y;
        std::tie(x, y) = top_left_xy;

        int get_text_size_font_thickness = (outline_color_rgb == cv::Scalar(0, 0, 0)) ? font_thickness : OUTLINE_FONT_THICKNESS;

        cv::Size text_size = cv::getTextSize(line, font_face, font_scale, get_text_size_font_thickness, nullptr);
        int line_height_no_baseline = text_size.height;
        int baseline = 0;

        int line_height = line_height_no_baseline + baseline;

        if (!bg_color_rgb.empty() && !line.empty()) {
            int sz_h = std::min(im_h - y, line_height);
            int sz_w = std::min(im_w - x, text_size.width);

            if (sz_h > 0 && sz_w > 0) {
                cv::Mat bg_mask(sz_h, sz_w, CV_8UC3, bg_color_rgb);
                bg_mask.copyTo(image_rgb(cv::Rect(x, y, sz_w, sz_h)));
            }
        }

        if (!outline_color_rgb.empty()) {
            cv::putText(
                image_rgb,
                line,
                cv::Point(x, y + line_height_no_baseline),
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv::LINE_AA
            );
        }

        cv::putText(
            image_rgb,
            line,
            cv::Point(x, y + line_height_no_baseline),
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv::LINE_AA
        );

        top_left_xy = std::make_tuple(x, y + static_cast<int>(line_height * line_spacing));
    }
}

cv::Mat drawBottom(
    cv::Mat bev,
    const std::vector<cv::Point>& rectangle,
    const std::vector<cv::Point>& ps_bev,
    const cv::Scalar& color = cv::Scalar(128, 128, 128),
    int thickness = 35
) {
    for (int i = 0; i < 4; ++i) {
        cv::line(bev, ps_bev[i], ps_bev[(i + 1) % 4], color, thickness);
    }

    return bev;
}

void drawSegments(
    cv::Mat& flow,
    const std::vector<std::vector<cv::Point>>& segments,
    const cv::Scalar& color = cv::Scalar(0, 0, 0)
) {
    for (const std::vector<cv::Point>& s : segments) {
        std::vector<std::vector<cv::Point>> contour_vec;
        contour_vec.push_back(s);
        cv::drawContours(flow, contour_vec, -1, color, -1);
    }
}

void drawAll(
    cv::Mat& frame_debug,
    cv::Mat& rgb,
    const std::vector<cv::Point>& best_lower_face,
    const std::vector<cv::Point>& best_upper_face,
    cv::Mat& bev,
    const std::vector<cv::Point>& best_rectangle,
    const std::vector<cv::Point>& best_ps_bev,
    const std::vector<std::vector<cv::Point>>& best_segments,
    bool final = true
) {
    // You can implement the draw_cube and other functions here as needed.
}

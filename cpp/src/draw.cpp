#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <spdlog/spdlog.h>
#include "draw.hpp"
#include "segment.hpp"

#include <spdlog/spdlog.h>

extern std::shared_ptr<spdlog::logger> logger;

void draw_image(const std::string &title, const cv::Mat &image)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, image);
    cv::waitKey(0);  // Wait for a key press or use a specific time delay
    cv::destroyAllWindows();
}

cv::Mat draw_bottom(
    cv::Mat bev,
    const std::vector<cv::Point> &rectangle,
    const std::vector<cv::Point> &ps_bev,
    const cv::Scalar &color = cv::Scalar(128, 128, 128),
    int thickness = 35
)
{
    for (int i = 0; i < 4; ++i)
    {
        cv::line(bev, ps_bev[i], ps_bev[(i + 1) % 4], color, thickness);
    }
    return bev;
}

void draw_segments(
    cv::Mat &flow,
    const std::vector<std::vector<cv::Point>> &segments,
    const cv::Scalar &color = cv::Scalar(0, 0, 0)
)
{
    for (const std::vector<cv::Point> &s : segments)
    {
        std::vector<std::vector<cv::Point>> contour_vec;
        contour_vec.push_back(s);
        cv::drawContours(flow, contour_vec, -1, color, -1);
    }
}

cv::Mat draw_cube(cv::Mat im, const std::vector<cv::Point2f> &lower_face,
                  const std::vector<cv::Point2f> &upper_face, cv::Scalar color, int lw)
{
    logger->info("{}", __func__);
    if (lower_face.empty() || upper_face.empty())
    {
        return im;
    }
    for (int i = 0; i < 4; ++i)
    {
        cv::line(im, lower_face[i], lower_face[(i + 1) % 4], color, lw);
        cv::line(im, upper_face[i], upper_face[(i + 1) % 4], color, lw);
        cv::line(im, lower_face[i], upper_face[i], color, lw);
    }
    return im;
}

cv::Mat plot_best_segments_simple(
    cv::Mat frame,
    cv::Mat bev,
    Forest &forest,
    double min_score
)
{
    logger->info("{}", __func__);
    static int count = 0;
    count++;
    int width = frame.cols;
    int height = frame.rows;
    cv::Mat seg = frame.clone();
    cv::Mat frame_copy = frame.clone();
    cv::Mat frame_orig = frame.clone();
    cv::Mat bev_copy = bev.clone();
    std::vector<SegmentData> best_segments = forest.get_best_segments();
    logger->info("Segmens Number: {}", best_segments.size());
    for (const SegmentData &segment_data : best_segments)
    {
        const std::set<int> &segment = segment_data.seg;
        double score = segment_data.score;
        logger->info("Segment Score: {}", score);
        Solution solution = segment_data.sol;
        double move = segment_data.move;
        if (score > min_score)
        {
            cv::Vec3b color;
            if (solution.cls == 0)
            {
                color = cv::Vec3b(0, 255, 255);  // Yellow
            }
            else if (solution.cls == 1)
            {
                color = cv::Vec3b(0, 255, 0);  // Green
            }
            else if (solution.cls == 2)
            {
                color = cv::Vec3b(0, 255, 255);  // Mint
            }
            for (int node_id : segment)
            {
                int x = node_id % width;
                int y = node_id / width;
                seg.at<cv::Vec3b>(y, x) = color;
            }
            color = cv::Vec3b(255, 0, 0);  // Blue
            draw_cube(frame, solution.lower_face, solution.upper_face, color, 1);
            draw_cube(seg, solution.lower_face, solution.upper_face, color, 1);
        }
        else
        {
            logger->warn("Low Segment Score: {} < {}", score, min_score);
        }
    }
    double opacity = 2.0 / 5.0;
    cv::addWeighted(frame, 1.0 - opacity, seg, opacity, 0, frame);
    return frame;
}

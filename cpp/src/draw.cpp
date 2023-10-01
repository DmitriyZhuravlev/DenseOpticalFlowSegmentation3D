#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include "draw.hpp"
#include "segment.hpp"

void draw_image(const std::string &title, const cv::Mat &image)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, image);
    cv::waitKey(0);  // Wait for a key press or use a specific time delay
    cv::destroyAllWindows();
}

cv::Mat drawBottom(
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

void drawSegments(
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

cv::Mat drawCube(cv::Mat im, const std::vector<cv::Point2f> &lowerFace,
                 const std::vector<cv::Point2f> &upperFace, cv::Scalar color, int lw)
{
    if (lowerFace.empty() || upperFace.empty())
    {
        return im;
    }

    for (int i = 0; i < 4; ++i)
    {
        cv::line(im, lowerFace[i], lowerFace[(i + 1) % 4], color, lw);
        cv::line(im, upperFace[i], upperFace[(i + 1) % 4], color, lw);
        cv::line(im, lowerFace[i], upperFace[i], color, lw);
    }

    return im;
}

cv::Mat plotBestSegmentsSimple(
    cv::Mat frame,
    cv::Mat bev,
    Forest &forest,
    double minScore
)
{
    static int count = 0;
    count++;

    int width = frame.cols;
    int height = frame.rows;

    cv::Mat seg = frame.clone();
    cv::Mat frameCopy = frame.clone();
    cv::Mat frameOrig = frame.clone();
    cv::Mat bevCopy = bev.clone();

    cv::Mat lf = (cv::Mat_<uint16_t>(4, 2) << 215, 265, 90, 121, 294, 120, 625, 265);
    cv::Mat uf = (cv::Mat_<uint16_t>(4, 2) << 215, 185, 90, 85, 294, 85, 625, 185);

    std::string outputFilePath = "output_frames/frame_" + std::to_string(count) + ".jpg";

    std::vector<SegmentData> bestSegments = forest.GetBestSegments();

    for (const SegmentData &segmentData : bestSegments)
    {
        std::vector<int> segment = segmentData.seg;
        double score = segmentData.score;
        Solution solution = segmentData.sol;
        double move = segmentData.move;

        if (score > minScore)
        {
            cv::Vec3b color;

            if (solution.cls == 0)
            {
                color = cv::Vec3b(0, 255, 255); // Yellow
            }
            else if (solution.cls == 1)
            {
                color = cv::Vec3b(0, 255, 0); // Green
            }
            else if (solution.cls == 2)
            {
                color = cv::Vec3b(0, 255, 255); // Mint
            }

            for (int node_id : segment)
            {
                int x = node_id % width;
                int y = node_id / width;
                seg.at<cv::Vec3b>(y, x) = color;
            }

            color = cv::Vec3b(255, 0, 0); // Blue
            drawCube(frame, solution.lower_face, solution.upper_face, color, 1);
            drawCube(seg, solution.lower_face, solution.upper_face, color, 1);
        }
    }

    double opacity = 2.0 / 5.0;
    cv::addWeighted(frame, 1.0 - opacity, seg, opacity, 0, frame);

    return frame;
}

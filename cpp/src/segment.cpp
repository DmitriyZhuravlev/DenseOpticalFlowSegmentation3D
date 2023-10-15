#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "segment.hpp"
#include "lifting_3d.hpp"
#include "draw.hpp"

using namespace cv;
using namespace std;

bool debug = false;

std::shared_ptr<spdlog::logger> logger;

double diff(const Mat &flow, int x1, int y1, int x2, int y2) {
    cv::Point2f flow1 = flow.at<cv::Point2f>(y1, x1);
    cv::Point2f flow2 = flow.at<cv::Point2f>(y2, x2);

    // Calculate the difference between the x and y components of the vectors
    double delta_x = flow1.x - flow2.x;
    double delta_y = flow1.y - flow2.y;

    // Calculate the Euclidean distance between the vectors
    double distance = std::sqrt(delta_x * delta_x + delta_y * delta_y);

    return distance;
}

Forest get_segmented_array(const cv::Mat &flow, const cv::Mat &bev, const cv::Matx33f &persp_mat,
                           const cv::Matx33f &inv_mat,
                           const std::vector<cv::Matx33f> &inv_mat_upper, int neighbor = 8) {
    // Check for valid neighborhood value (4 or 8)
    if (neighbor != 4 && neighbor != 8) {
        // Log a warning if the neighborhood is invalid
        logger->error("Invalid neighborhood chosen. The acceptable values are 4 or 8.");
        logger->error("Segmenting with 4-neighborhood...");
        // TODO add throw
    }

    // Record the start time
    double start_time = static_cast<double>(cv::getTickCount());

    int height = flow.rows;
    int width = flow.cols;

    // Apply Gaussian blur to the flow image
    cv::GaussianBlur(flow, flow, cv::Size(0, 0), 3.0);

    // Create the graph edges
    const std::vector<Edge> graph_edges = build_graph(flow, width, height, diff, neighbor == 8);
    logger->info("Edges number: {}", graph_edges.size());

    // Segment the graph using the forest and sorted graph
    //Forest forest(flow, bev, persp_mat, inv_mat, inv_mat_upper);

    //std::vector<Edge> sorted_graph;
    auto merged_forest = segment_graph(flow, graph_edges, bev, persp_mat, inv_mat,
                                       inv_mat_upper);
    //merged_forest.LogInfo();
    // Calculate elapsed time
    double elapsed_time = (static_cast<double>(cv::getTickCount()) - start_time) /
                          cv::getTickFrequency();

    // Additional code can be added here to work with the 'forest' object if needed

    return merged_forest;
}

void init_logger() {
    // Create a logger with the name "my_logger" and use the stdout color sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    logger = std::make_shared<spdlog::logger>("seg", console_sink);

    // Set the log level (optional)
    logger->set_level(spdlog::level::err); //info); //err); // You can change the log level as needed

    // Register the logger globally
    spdlog::register_logger(logger);
}

int main() {
    init_logger();

    // Load the consecutive images and convert them to grayscale
    cv::Mat im1 = cv::imread("/home/dzhura/ComputerVision/3dim-optical-flow/img/frame_1052.png");
    cv::Mat im2 = cv::imread("/home/dzhura/ComputerVision/3dim-optical-flow/img/frame_1053.png");

    cv::Mat gray1, gray2;
    cv::cvtColor(im1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(im2, gray2, cv::COLOR_BGR2GRAY);

    cv::Mat flow(gray1.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // Visualization
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    // Build HSV image
    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

    //cv::imshow("frame2", bgr);
    //cv::waitKey(0);

    // Call the get_mat function to obtain persp_mat and inv_mat
    //cv::Mat persp_mat, inv_mat;
    auto [persp_mat, inv_mat] =  get_mat();

    std::cout << persp_mat << std::endl;

    // Initialize inv_mat_upper
    std::vector<cv::Matx33f> inv_mat_upper(3);
    inv_mat_upper[0] = get_mat_upper(0);
    inv_mat_upper[1] = get_mat_upper(1);
    inv_mat_upper[2] = get_mat_upper(2);

    // Read the input image
    cv::Mat img_empty = cv::imread("/home/dzhura/ComputerVision/3dim-optical-flow/img/background.png");

    // Convert RGB to BGR for OpenCV
    cv::cvtColor(img_empty, img_empty, cv::COLOR_RGB2BGR);

    // Perform the transformation on another image or frame (frame variable not provided)
    // cv::Mat frame; // Assuming you have a frame variable defined
    cv::Mat bev = transform(im2, persp_mat); // Call your transform function

    //cv::imshow("frame2", bev);
    //cv::waitKey(0);

    //cv::Mat bev_out;
    //cv::addWeighted(bev, 0.5, background_bev, 0.5, 0.5, bev_out);

    // Create a forest object and get segmented array
    Forest forest = get_segmented_array(flow, bev, persp_mat, inv_mat, inv_mat_upper, 8);

    // Get the best segments from the forest
    //std::vector<SegmentData> best_segments = forest.GetBestSegments();

    // Call the plot_best_segments_simple function
    cv::Mat new_segments = plot_best_segments_simple(im2, bev, forest, 0.7);

    // Assuming you have a function named `draw_image` to display the new segments
    draw_image("New Segments", new_segments);

    return 0;
}

int main1() {
    init_logger();

    //cv::Mat persp_mat, inv_mat;
    auto [persp_mat, inv_mat] =  get_mat();

    //std::cout << persp_mat << std::endl;

    // Initialize inv_mat_upper
    std::vector<cv::Matx33f> inv_mat_upper(3);
    inv_mat_upper[0] = get_mat_upper(0);
    inv_mat_upper[1] = get_mat_upper(1);
    inv_mat_upper[2] = get_mat_upper(2);

    // Read the input image
    cv::Mat img_empty = cv::imread("/home/dzhura/ComputerVision/3dim-optical-flow/img/background.png");

    // Convert RGB to BGR for OpenCV
    cv::cvtColor(img_empty, img_empty, cv::COLOR_RGB2BGR);

    // Perform the transformation on another image or frame (frame variable not provided)
    // cv::Mat frame; // Assuming you have a frame variable defined
    cv::Mat bev = transform(img_empty, persp_mat); // Call your transform function

    // Open the video file for reading
    cv::VideoCapture video("/home/dzhura/ComputerVision/3dim-optical-flow/video/video_3.mp4");

    if (!video.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    cv::Mat frame, prev_frame;
    cv::Mat flow;

    while (true) {
        video >> frame;  // Read the next frame

        if (frame.empty()) {
            break;  // End of video
        }

        if (prev_frame.empty()) {
            prev_frame = frame.clone();
            continue;
        }

        cv::Mat gray1, gray2;
        cv::cvtColor(prev_frame, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, gray2, cv::COLOR_BGR2GRAY);

        // Compute optical flow between the two frames
        cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // Visualization
        cv::Mat flow_parts[2];
        cv::split(flow, flow_parts);
        cv::Mat magnitude, angle, magn_norm;
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        // Build HSV image
        cv::Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        cv::merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

        // Further processing logic goes here...

        // Display the frame with optical flow

        // Create a forest object and get segmented array
        Forest forest = get_segmented_array(flow, bev, persp_mat, inv_mat, inv_mat_upper, 8);

        // Get the best segments from the forest
        //std::vector<SegmentData> best_segments = forest.GetBestSegments();

        // Call the plot_best_segments_simple function
        cv::Mat new_segments = plot_best_segments_simple(frame, bev, forest, 0.7);

        // Assuming you have a function named `draw_image` to display the new segments
        cv::imshow("Segmented Optical Flow", new_segments);
        //draw_image("New Segments", new_segments);

        int key = cv::waitKey(30);
        if (key == 27)    // Press 'Esc' to exit
        {
            break;
        }

        prev_frame = frame.clone();
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}

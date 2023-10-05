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

vector<pair<Point2f, Point2f>> buildGraph(const Mat &smooth, int width, int height,
                            const function<float(const Mat &, int, int, int, int)> &diffFunc, bool is8Connected)
{
    vector<pair<Point2f, Point2f>> graphEdges;
    graphEdges.reserve(is8Connected ? (width - 1) * (height - 1) * 8 : (width - 1) * (height - 1) * 4);

    for (int y = 0; y < height - 1; ++y)
    {
        for (int x = 0; x < width - 1; ++x)
        {
            Point2f source(static_cast<float>(x), static_cast<float>(y));
            Point2f right(static_cast<float>(x + 1), static_cast<float>(y));
            Point2f down(static_cast<float>(x), static_cast<float>(y + 1));
            Point2f rightDown(static_cast<float>(x + 1), static_cast<float>(y + 1));

            float weight1 = diffFunc(smooth, x, y, x + 1, y);
            float weight2 = diffFunc(smooth, x, y, x, y + 1);
            graphEdges.emplace_back(source, right);
            graphEdges.emplace_back(source, down);
            if (is8Connected)
            {
                float weight3 = diffFunc(smooth, x, y, x + 1, y + 1);
                float weight4 = diffFunc(smooth, x + 1, y, x, y + 1);
                graphEdges.emplace_back(source, rightDown);
                graphEdges.emplace_back(right, down);
                graphEdges.emplace_back(right, rightDown);
                graphEdges.emplace_back(down, rightDown);
                graphEdges.emplace_back(source, rightDown);
            }
        }
    }

    return graphEdges;
}

std::pair<Forest, std::vector<std::tuple<int, int, double>>> segmentGraph(
    const cv::Mat &flow,
    const std::vector<std::tuple<int, int, double>> &graphEdges,
    const cv::Mat &bev,
    const cv::Matx33f &perspMat,
    const cv::Matx33f &invMat,
    const std::vector<cv::Matx33f> &invMatUpper
)
{
    Forest forest(flow, bev, perspMat, invMat, invMatUpper);

    auto weight = [](const std::tuple<int, int, double> &edge) -> double
    {
        return std::get<2>(edge);
    };

    std::vector<std::tuple<int, int, double>> sortedGraph = graphEdges;
    std::sort(sortedGraph.begin(), sortedGraph.end(), [weight](const auto & edge1, const auto & edge2)
    {
        return weight(edge1) < weight(edge2);
    });

    for (const auto &edge : sortedGraph)
    {
        int a = forest.find(std::get<0>(edge));
        int b = forest.find(std::get<1>(edge));

        if (a == b)
        {
            continue;
        }

        forest.new_merge(a, b);
    }

    return std::make_pair(forest, sortedGraph);
}

double diff(const Mat &img, int x1, int y1, int x2, int y2)
{
    double out = 0.0f;
    for (int c = 0; c < img.channels(); ++c)
    {
        out += static_cast<double>(pow(img.at<Vec3b>(y1, x1)[c] - img.at<Vec3b>(y2, x2)[c], 2));
    }
    return out;
}

float threshold(int size, float constant)
{
    return constant / static_cast<float>(size);
}

Mat generateImage(const Forest &forest, int width, int height, float threshold,
                  const Mat &inputImage)
{
    vector<Vec3b> colors(width * height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int node = y * width + x;
            int root = forest.find(node);

            if (diff(inputImage, x, y, x, y) > threshold)
            {
                colors[root] = inputImage.at<Vec3b>(y, x);
            }
        }
    }

    Mat image(height, width, CV_8UC3);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int node = y * width + x;
            int root = forest.find(node);
            image.at<Vec3b>(y, x) = colors[root];
        }
    }

    return image;
}

//std::tuple<double, Solution> get_score(const std::tuple<int, int, int, int> &bbox,
//const cv::Vec2f &direction,
//const std::vector<std::pair<double, double>>, const cv::Mat &persp_mat, const cv::Mat &inv_mat,
//const std::vector<cv::Mat> &inv_mat_upper)
//{
//double max_score = -1.0;
////Solution(int cls, const std::vector<std::pair<double, double>> &ps_bev, const std::vector<std::pair<double, double>> &lower_face,
////const std::vector<std::pair<double, double>> &upper_face,
////const std::vector<std::pair<double, double>> &rectangle, double w_error, double h_error, double orient)
//Solution sol; //(0, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), 0.0, 0.0, 0.0);

//for (int cls = 0; cls < 3; ++cls)
//{
//std::vector<std::pair<double, double>> ps_bev, lower_face, upper_face, rectangle;
//double w_error, h_error, deg;

//// Call get_bottom_variants function to obtain ps_bev, lower_face, upper_face, rectangle, w_error, h_error, and deg
//// You will need to provide the appropriate parameters for this function call.

//if (!rectangle.empty() && max_score < (w_error + h_error) / 2)
//{
//max_score = (w_error + h_error) / 2;
//sol = Solution(cls, ps_bev, lower_face, upper_face, rectangle, w_error, h_error, deg);
//}
//}

//return std::make_tuple(max_score, sol);
//}

Forest get_segmented_array(cv::Mat flow, cv::Mat bev, cv::Matx33f persp_mat, cv::Matx33f inv_mat,
                           std::vector<cv::Matx33f> inv_mat_upper, int neighbor = 8)
{
    // Check for valid neighborhood value (4 or 8)
    if (neighbor != 4 && neighbor != 8)
    {
        // Log a warning if the neighborhood is invalid
        std::cerr << "Invalid neighborhood chosen. The acceptable values are 4 or 8." << std::endl;
        std::cerr << "Segmenting with 4-neighborhood..." << std::endl;
    }

    // Record the start time
    double start_time = static_cast<double>(cv::getTickCount());

    int height = flow.rows;
    int width = flow.cols;

    // Apply Gaussian blur to the flow image
    cv::GaussianBlur(flow, flow, cv::Size(0, 0), 3.5);

    // Create the graph edges
    std::vector<Edge> graph_edges = build_graph(flow, width, height, diff, neighbor == 8);
    logger->info("Edges number: {}", graph_edges.size());

    // Segment the graph using the forest and sorted graph
    //Forest forest(flow, bev, persp_mat, inv_mat, inv_mat_upper);

    //std::vector<Edge> sorted_graph;
    auto [merged_forest, sorted_graph] = segment_graph(flow, graph_edges, bev, persp_mat, inv_mat,
                                         inv_mat_upper);
    merged_forest.LogInfo();
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
    logger->set_level(spdlog::level::info); // You can change the log level as needed
    
    // Register the logger globally
    spdlog::register_logger(logger);
}

int main()
{

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

    //cv::imshow("frame2", img_empty);
    //cv::waitKey(0);

    // Perform perspective transformation
    //cv::Mat background_bev;
    //cv::warpPerspective(img_empty, background_bev, persp_mat, img_empty.size(), cv::INTER_LINEAR,
                        //cv::BORDER_CONSTANT);

    //// Save the transformed image
    //cv::imwrite("/home/dzhura/ComputerVision/3dim-optical-flow/img/background_bev.png", background_bev);

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
    cv::Mat new_segments = plotBestSegmentsSimple(im2, bev, forest, 0.5);

    // Assuming you have a function named `plot_image` to display the new segments
    draw_image("New Segments", new_segments);

    return 0;
}

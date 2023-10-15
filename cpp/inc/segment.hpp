#ifndef SEGMENT_HPP
#define SEGMENT_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <functional>
#include "graph.hpp"

extern bool debug;

//class Solution
//{
//public:
//int cls;
//std::vector<std::pair<double, double>> ps_bev;
//std::vector<std::pair<double, double>> lower_face;
//std::vector<std::pair<double, double>> upper_face;
//std::vector<std::pair<double, double>> rectangle;
//double w_error;
//double h_error;
//double orient;

//// Default constructor
//Solution()
//: cls(0), w_error(0.0), h_error(0.0), orient(0.0) {}

//Solution(int cls, const std::vector<std::pair<double, double>> &ps_bev, const std::vector<std::pair<double, double>> &lower_face,
//const std::vector<std::pair<double, double>> &upper_face,
//const std::vector<std::pair<double, double>> &rectangle, double w_error, double h_error, double orient)
//: cls(cls), ps_bev(ps_bev), lower_face(lower_face), upper_face(upper_face), rectangle(rectangle),
//w_error(w_error), h_error(h_error), orient(orient) {}
//};

//class SegmentData
//{
//public:
//double score;              // Type: double
//std::vector<int> seg;    // Type: std::vector<int>
//Solution sol;            // Type: Solution (Assuming you have defined Solution)
//double move;              // Type: double

//SegmentData(double &score, std::vector<int> seg, Solution &sol, double &move);
//};

//std::vector<std::pair<cv::Point2f, cv::Point2f>> buildGraph(const cv::Mat &smooth, int width,
        //int height,
        //const std::function<float(const cv::Mat &, int, int, int, int)> &diffFunc,
        //bool is8Connected);

//std::pair<Forest, std::vector<std::tuple<int, int, double>>> segmentGraph(
    //const cv::Mat &flow,
    //const std::vector<std::tuple<int, int, double>> &graphEdges,
    //const cv::Mat &bev,
    //const cv::Mat &perspMat,
    //const cv::Mat &invMat,
    //const std::vector<cv::Mat> &invMatUpper
//);

//float threshold(int size, float constant);

//cv::Mat generateImage(const DisjointSetForest& forest, int width, int height, float threshold, const cv::Mat& inputImage);

//std::tuple<double, Solution> get_score(const std::tuple<int, int, int, int> &bbox,
//const cv::Vec2f &direction, const cv::Mat &bev, const cv::Mat &persp_mat, const cv::Mat &inv_mat,
//const std::vector<cv::Mat> &inv_mat_upper);
#endif // SEGMENT_HPP

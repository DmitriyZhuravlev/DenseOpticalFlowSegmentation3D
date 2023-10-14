#ifndef DRAW_HPP
#define DRAW_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include "graph.hpp"

void draw_image(const std::string &title, const cv::Mat &image);
//cv::Mat drawBottom(cv::Mat image, const std::vector<cv::Point_<int>>& bottom, const std::vector<cv::Point_<int>>& bottom_out, const cv::Scalar& color, int thickness);
//void drawSegments(cv::Mat& image, const std::vector<std::vector<cv::Point_<int>>>& segments, const cv::Scalar& color);
cv::Mat plotBestSegmentsSimple(cv::Mat frame, cv::Mat seg, Forest &forest, double seg_thr = 0.6);
//cv::Mat drawCube(cv::Mat im, std::vector<cv::Point> lowerFace, std::vector<cv::Point> upperFace, cv::Scalar color = cv::Scalar(0, 0, 255), bool bgr = false, int lw = 2);

#endif // DRAW_HPP

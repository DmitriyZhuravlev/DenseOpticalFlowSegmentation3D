#ifndef LIFTING_3D_HPP
#define LIFTING_3D_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <deque>
#include <set>
#include "graph.hpp"

Solution get_bottom_variants(const cv::Mat &orig_mov_dir, const std::vector<int> &box_2d,
                             const cv::Mat &mat, const cv::Mat &inv_mat,
                             const std::vector<cv::Mat> &inv_matrix_upper, int cls, bool debug = false);
std::pair<cv::Matx33f, cv::Matx33f> get_mat();
cv::Matx33f get_mat_upper(int cls);
cv::Mat transform(cv::Mat img, cv::Matx33f mat);
#endif

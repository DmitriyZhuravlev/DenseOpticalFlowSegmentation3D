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

Solution get_bottom_variants(const cv::Point2f &orig_mov_dir,
                             const std::vector<cv::Point2i> &box_2d,
                             const cv::Matx33f &mat, const cv::Matx33f &inv_mat,
                             const cv::Matx33f &inv_matrix_upper, int cls);
std::pair<cv::Matx33f, cv::Matx33f> get_mat();
cv::Matx33f get_mat_upper(int cls);
cv::Mat transform(cv::Mat img, cv::Matx33f mat);

std::vector<cv::Point2f> get_upper_face(const std::vector<cv::Point2i> &box_2d,
                                        const std::vector<cv::Point2f> &lower_face);
std::vector<cv::Point2f> get_upper_face_simple(std::vector<cv::Point2i> box_2d,
        std::vector<cv::Point2f> lower_face);
std::pair<double, double> get_obj_size(int cls);
cv::Point2f get_intersect(cv::Point2f a1, cv::Point2f a2, cv::Point2f b1, cv::Point2f b2);
#endif

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <deque>
#include <set>
//#include "segment.hpp"

struct Edge
{
    int start;
    int end;
    double weight;
};

// Define a function signature for the 'diff' function, assuming it takes two pairs of (x, y) coordinates
typedef double (*DiffFunction)(const cv::Mat &flow, int x0, int y0, int x1, int y1);
// Define the function signatures
Edge create_edge(const cv::Mat &flow, int width, int x, int y, int x1, int y1,
                 const DiffFunction &diff);
std::vector<Edge> build_graph(const cv::Mat &img, int width, int height, const DiffFunction &diff,
                              bool neighborhood_8 = false);

class Solution
{
    public:
        int cls;
        std::vector<cv::Point2f>ps_bev;
        std::vector<cv::Point2f>lower_face;
        std::vector<cv::Point2f>upper_face;
        std::vector<cv::Point2f>rectangle;
        double w_error;
        double h_error;
        double orient;

        // Default constructor
        Solution()
            : w_error(-1.0), h_error(-1.0) {}

        Solution(int cls, const std::vector<cv::Point2f> &ps_bev,
                 const std::vector<cv::Point2f> &lower_face,
                 const std::vector<cv::Point2f> &upper_face,
                 const std::vector<cv::Point2f> &rectangle, double w_error, double h_error, double orient)
            : cls(cls), ps_bev(ps_bev), lower_face(lower_face), upper_face(upper_face), rectangle(rectangle),
              w_error(w_error), h_error(h_error), orient(orient) {}
};

class SegmentData
{
    public:
        double score;              // Type: double
        std::vector<int> seg;    // Type: std::vector<int>
        Solution sol;            // Type: Solution (Assuming you have defined Solution)
        double move;              // Type: double

        SegmentData(double &score, std::vector<int> seg, Solution &sol, double &move);
};

class Node
{
    public:
        int id;
        int parent;
        int rank;
        int size;
        cv::Vec2f flow_value;

        Node(int id, const std::complex<float> &complex_value);

        friend std::ostream &operator<<(std::ostream &os, const Node &node);
};

class Forest
{
    public:
        int num_sets;
        int width;
        int height;
        int min_move;
        cv::Mat bev;
        cv::Matx33f persp_mat;
        cv::Matx33f inv_mat;
        std::vector<cv::Matx33f> inv_mat_upper;

        Forest(const cv::Mat &flow, const cv::Mat &bev, const cv::Matx33f &persp_mat,
               const cv::Matx33f &inv_mat,
               const std::vector<cv::Matx33f> &inv_mat_upper, int min_move = 5);

        int find(int n);
        int find(int n) const;

        int merge(int a, int b);

        void new_merge(int a, int b, double score_threshold = 0.5, int min_size = 1000,
                       double min_move = 1.0, double min_convexity = 1.0 / 2.0);

        double get_segment_best_score(int node_id) const;
        void LogInfo() const;

        std::vector<SegmentData> GetBestSegments();

        std::vector<std::set<int>> get_segments();

        std::vector<cv::Point2i> get_bounding_box(int node_id) const;

    private:
        std::vector<Node> nodes;
        std::vector<std::set<int>> segments;
        std::vector<std::vector<SegmentData>> segment_history;
        std::vector<double> segment_scores;

        // Include any other private member functions or variables here
};

//std::tuple<double, Solution> get_score(const std::tuple<int, int, int, int> &bbox,
                                       //const cv::Vec2f &direction, const cv::Mat &bev, const cv::Mat &persp_mat, const cv::Mat &inv_mat,
                                       //const std::vector<cv::Mat> &inv_mat_upper);

std::pair<Forest, std::vector<Edge>> segment_graph(const cv::Mat &flow,
                                  const std::vector<Edge> &graph_edges, const cv::Mat &bev, const cv::Matx33f &persp_mat,
                                  const cv::Matx33f &inv_mat, const std::vector<cv::Matx33f> &inv_mat_upper);

#endif

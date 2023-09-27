#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <deque>

class Node {
public:
    int id;
    int parent;
    int rank;
    int size;
    cv::Mat flow_value;

    Node(int id, const cv::Mat& flow_value) : id(id), parent(id), rank(0), size(1), flow_value(flow_value) {}

    friend std::ostream& operator<<(std::ostream& os, const Node& node) {
        os << "(parent=" << node.parent << ", rank=" << node.rank << ", size=" << node.size << ")";
        return os;
    }
};

class Forest {
public:
    int num_sets;
    int width;
    int height;
    int min_move;
    cv::Mat bev;
    cv::Mat persp_mat;
    cv::Mat inv_mat;
    std::vector<cv::Mat> inv_mat_upper;

    Forest(const cv::Mat& flow, const cv::Mat& bev, const cv::Mat& persp_mat, const cv::Mat& inv_mat, 
           const std::vector<cv::Mat>& inv_mat_upper, int min_move = 5) 
           : num_sets(flow.rows * flow.cols), width(flow.cols), height(flow.rows), min_move(min_move), bev(bev), 
           persp_mat(persp_mat), inv_mat(inv_mat), inv_mat_upper(inv_mat_upper) {
        nodes.reserve(flow.rows * flow.cols);
        segments.reserve(flow.rows * flow.cols);
        segment_history.resize(flow.rows * flow.cols);
        
        for (int i = 0; i < flow.rows * flow.cols; ++i) {
            nodes.emplace_back(i, flow.at<cv::Vec3f>(i / flow.cols, i % flow.cols));
            segments.emplace_back(std::set<int>{i});
        }
    }

    int find(int n) {
        if (n != nodes[n].parent) {
            nodes[n].parent = find(nodes[n].parent);
        }
        return nodes[n].parent;
    }

    void merge(int a, int b) {
        int parent_a = find(a);
        int parent_b = find(b);

        if (parent_a != parent_b) {
            if (nodes[parent_a].rank > nodes[parent_b].rank) {
                std::swap(parent_a, parent_b);
            }

            nodes[parent_a].parent = parent_b;

            int size_a = nodes[parent_a].size;
            int size_b = nodes[parent_b].size;
            cv::Mat weighted_flow_a = nodes[parent_a].flow_value * size_a;
            cv::Mat weighted_flow_b = nodes[parent_b].flow_value * size_b;
            cv::Mat avg_flow = (weighted_flow_a + weighted_flow_b) / (size_a + size_b);

            nodes[parent_b].flow_value = avg_flow;

            segments[parent_b].insert(segments[parent_a].begin(), segments[parent_a].end());
            segments[parent_a].clear();
            nodes[parent_b].size += size_a;
            nodes[parent_a].size = 0;

            if (nodes[parent_a].rank == nodes[parent_b].rank) {
                nodes[parent_b].rank += 1;
            }

            num_sets -= 1;
        }
    }

    void new_merge(int a, int b, double score_threshold = 0.5, int min_size = 1000, double min_move = 1.0, double min_convexity = 1.0 / 2.0) {
        int parent_b = find(b);

        if (nodes[parent_b].size < min_size) {
            return;
        }
        
        int x = parent_b % width;
        int y = parent_b / width;
        if (y < height / 10) {
            return;
        }

        double move = cv::norm(nodes[parent_b].flow_value);

        if (move < (y + 1) / static_cast<double>(height)) {
            return;
        }

        std::tuple<int, int, int, int> bbox = get_bounding_box(parent_b);
        double convexity = nodes[parent_b].size / ((std::get<2>(bbox) - std::get<0>(bbox) + 1) * (std::get<3>(bbox) - std::get<1>(bbox) + 1));

        std::tuple<double, Solution> score_solution = get_score(bbox, nodes[parent_b].flow_value, bev, persp_mat, inv_mat, inv_mat_upper);
        double score = std::get<0>(score_solution);
        Solution solution = std::get<1>(score_solution);
        
        segment_scores[parent_b] = score;

        if (solution.cls == 0) {
            min_convexity = 3.0 / 4.0;
        }
        if (solution.cls == 1) {
            min_convexity = 1.0 / 2.0;
        }
        if (solution.cls == 2) {
            min_convexity = 20.0 / 29.0;
        }

        if (convexity < min_convexity) {
            return;
        }

        if (score > score_threshold) {
            std::set<int>& seg = segments[parent_b];
            segment_history[parent_b].emplace_back(score, std::vector<int>(seg.begin(), seg.end()), solution, move);
        }
    }

    double get_segment_best_score(int node_id) const {
        return segment_scores[node_id];
    }

    std::vector<std::map<std::string, double>> get_best_segments() {
        std::vector<std::map<std::string, double>> best_segments;
        auto key_function = [](const auto& item) {
            double score = std::get<0>(item);
            const auto& segment = std::get<1>(item);
            return std::make_tuple(score, segment.size());
        };

        for (size_t segment_id = 0; segment_id < segment_history.size(); ++segment_id) {
            const auto& history = segment_history[segment_id];
            if (history.empty()) {
                continue;
            }

            auto max_element = std::max_element(history.begin(), history.end(), key_function);
            double best_score = std::get<0>(*max_element);
            const auto& best_segment = std::get<1>(*max_element);
            const auto& best_solution = std::get<2>(*max_element);
            double best_move = std::get<3>(*max_element);

            if (best_segments.find(segment_id) == best_segments.end() || best_score > best_segments[segment_id]["score"]) {
                best_segments[segment_id] = {
                    {"segment", best_segment},
                    {"score", best_score},
                    {"solution", best_solution.cls},
                    {"move", best_move}
                };
            }
        }

        return best_segments;
    }

    std::vector<std::set<int>> get_segments() {
        std::vector<std::set<int>> segments;
        for (size_t i = 0; i < nodes.size(); ++i) {
            int root = find(i);
            if (root >= segments.size()) {
                segments.resize(root + 1);
            }
            segments[root].insert(i);
        }
        return segments;
    }

    std::tuple<int, int, int, int> get_bounding_box(int node_id) const {
        const std::set<int>& segment = segments[find(node_id)];
        std::vector<int> x_values, y_values;
        x_values.reserve(segment.size());
        y_values.reserve(segment.size());

        for (int node_id : segment) {
            x_values.push_back(node_id % width);
            y_values.push_back(node_id / width);
        }

        int min_x = *std::min_element(x_values.begin(), x_values.end());
        int max_x = *std::max_element(x_values.begin(), x_values.end());
        int min_y = *std::min_element(y_values.begin(), y_values.end());
        int max_y = *std::max_element(y_values.begin(), y_values.end());

        return std::make_tuple(min_x, min_y, max_x, max_y);
    }
};

// Define other helper functions and the Solution class as needed.

int main() {
    // You can write the main function here to use the Forest and other classes.
    // Make sure to load the image, create a flow matrix, and initialize matrices like bev, persp_mat, and inv_mat.
    // Then, create Forest and call its methods as needed.
    return 0;
}

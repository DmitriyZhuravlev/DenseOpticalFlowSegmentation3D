#include "graph.hpp"
//#include "segment.hpp"

class Solution;

// Define the Edge type

Edge create_edge(const cv::Mat &flow, int width, int x, int y, int x1, int y1,
                 const DiffFunction &diff)
{
    auto vertex_id = [width](int x, int y) { return y * width + x; };
    double w = diff(flow, x, y, x1, y1);
    return {vertex_id(x, y), vertex_id(x1, y1), w};
}

std::vector<Edge> build_graph(const cv::Mat &img, int width, int height, const DiffFunction &diff,
                              bool neighborhood_8)
{
    std::vector<Edge> graph_edges;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (x > 0)
            {
                graph_edges.push_back(create_edge(img, width, x, y, x - 1, y, diff));
            }

            if (y > 0)
            {
                graph_edges.push_back(create_edge(img, width, x, y, x, y - 1, diff));
            }

            if (neighborhood_8)
            {
                if (x > 0 && y > 0)
                {
                    graph_edges.push_back(create_edge(img, width, x, y, x - 1, y - 1, diff));
                }

                if (x > 0 && y < height - 1)
                {
                    graph_edges.push_back(create_edge(img, width, x, y, x - 1, y + 1, diff));
                }
            }
        }
    }
    return graph_edges;
}

// Constructor definition
SegmentData::SegmentData(double &score, std::vector<int> seg, Solution &sol, double &move)
    : score(score), seg(seg), sol(sol), move(move)
{
    // Initialize member variables with provided values.
}

Node::Node(int id, const cv::Vec2f &flow_value) : id(id), parent(id), rank(0), size(1),
    flow_value(flow_value) {}

std::ostream &operator<<(std::ostream &os, const Node &node)
{
    os << "(parent=" << node.parent << ", rank=" << node.rank << ", size=" << node.size << ")";
    return os;
}

Forest::Forest(const cv::Mat &flow, const cv::Mat &bev, const cv::Matx33f &persp_mat,
               const cv::Matx33f &inv_mat,
               const std::vector<cv::Matx33f> &inv_mat_upper, int min_move)
    : num_sets(flow.rows * flow.cols), width(flow.cols), height(flow.rows), min_move(min_move),
      bev(bev),
      persp_mat(persp_mat), inv_mat(inv_mat), inv_mat_upper(inv_mat_upper)
{
    nodes.reserve(flow.rows * flow.cols);
    segments.reserve(flow.rows * flow.cols);
    segment_history.resize(flow.rows * flow.cols);
    segment_scores.resize(flow.rows * flow.cols);

    for (int i = 0; i < flow.rows * flow.cols; ++i)
    {
        nodes.emplace_back(i, flow.at<cv::Vec2f>(i / flow.cols, i % flow.cols));
        segments.emplace_back(std::set<int> {i});
    }
}

int Forest::find(int n)
{
    if (n != nodes[n].parent)
    {
        nodes[n].parent = find(nodes[n].parent);
    }
    return nodes[n].parent;
}

int Forest::find(int n) const
{
    // This version can be used on const objects but doesn't modify the state.
    int current = n;
    while (current != nodes[current].parent)
    {
        current = nodes[current].parent;
    }
    return current;
}

void Forest::merge(int a, int b)
{
    int parent_a = find(a);
    int parent_b = find(b);

    if (parent_a != parent_b)
    {
        if (nodes[parent_a].rank > nodes[parent_b].rank)
        {
            std::swap(parent_a, parent_b);
        }

        nodes[parent_a].parent = parent_b;

        int size_a = nodes[parent_a].size;
        int size_b = nodes[parent_b].size;
        cv::Vec2f weighted_flow_a = nodes[parent_a].flow_value * size_a;
        cv::Vec2f weighted_flow_b = nodes[parent_b].flow_value * size_b;
        cv::Vec2f avg_flow = (weighted_flow_a + weighted_flow_b) / (size_a + size_b);

        nodes[parent_b].flow_value = avg_flow;

        segments[parent_b].insert(segments[parent_a].begin(), segments[parent_a].end());
        segments[parent_a].clear();
        nodes[parent_b].size += size_a;
        nodes[parent_a].size = 0;

        if (nodes[parent_a].rank == nodes[parent_b].rank)
        {
            nodes[parent_b].rank += 1;
        }

        num_sets -= 1;
    }
}

std::tuple<double, Solution> get_score(const std::vector<cv::Point2i> &bbox,
                                       const cv::Vec2f &direction,
                                       const std::vector<cv::Point2f> bev, const cv::Matx33f &persp_mat, const cv::Matx33f &inv_mat,
                                       const std::vector<cv::Matx33f> &inv_mat_upper)
{
    double max_score = -1.0;
    //Solution(int cls, const std::vector<std::pair<double, double>> &ps_bev, const std::vector<std::pair<double, double>> &lower_face,
    //const std::vector<std::pair<double, double>> &upper_face,
    //const std::vector<std::pair<double, double>> &rectangle, double w_error, double h_error, double orient)
    Solution sol; //(0, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), 0.0, 0.0, 0.0);

    for (int cls = 0; cls < 3; ++cls)
    {
        std::vector<cv::Point2f> ps_bev, lower_face, upper_face, rectangle;
        double w_error, h_error, deg;

        // Call get_bottom_variants function to obtain ps_bev, lower_face, upper_face, rectangle, w_error, h_error, and deg
        // You will need to provide the appropriate parameters for this function call.

        if (!rectangle.empty() && max_score < (w_error + h_error) / 2)
        {
            max_score = (w_error + h_error) / 2;
            sol = Solution(cls, ps_bev, lower_face, upper_face, rectangle, w_error, h_error, deg);
        }
    }

    return std::make_tuple(max_score, sol);
}

void Forest::new_merge(int a, int b, double score_threshold, int min_size, double min_move,
                       double min_convexity)
{
    int parent_b = find(b);

    if (nodes[parent_b].size < min_size)
    {
        return;
    }

    int x = parent_b % width;
    int y = parent_b / width;
    if (y < height / 10)
    {
        return;
    }

    double move = cv::norm(nodes[parent_b].flow_value);

    if (move < (y + 1) / static_cast<double>(height))
    {
        return;
    }

    std::vector<cv::Point2i> bbox = get_bounding_box(parent_b);
    double convexity = nodes[parent_b].size / ((bbox[1].x - bbox[0].x + 1) *
                       (bbox[1].y - bbox[0].y + 1));

    std::tuple<double, Solution> score_solution = get_score(bbox, nodes[parent_b].flow_value, bev,
            persp_mat, inv_mat, inv_mat_upper);
    double score = std::get<0>(score_solution);
    Solution solution = std::get<1>(score_solution);

    segment_scores[parent_b] = score;

    if (solution.cls == 0)
    {
        min_convexity = 3.0 / 4.0;
    }
    if (solution.cls == 1)
    {
        min_convexity = 1.0 / 2.0;
    }
    if (solution.cls == 2)
    {
        min_convexity = 20.0 / 29.0;
    }

    if (convexity < min_convexity)
    {
        return;
    }

    if (score > score_threshold)
    {
        std::set<int> &seg = segments[parent_b];
        segment_history[parent_b].emplace_back(score, std::vector<int>(seg.begin(), seg.end()), solution,
                                               move);
    }
}

double Forest::get_segment_best_score(int node_id) const
{
    return segment_scores[node_id];
}

std::vector<SegmentData> Forest::GetBestSegments()
{
    std::vector<SegmentData> best_segments;
    auto key_function = [](const auto & item1, const auto & item2)
    {
        double score1 = item1.score;
        double score2 = item2.score;
        const auto &segment1 = item1.seg;
        const auto &segment2 = item2.seg;

        // Compare first by score and then by segment size if scores are equal
        if (score1 != score2)
        {
            return score1 < score2;
        }
        else
        {
            return segment1.size() < segment2.size();
        }
    };
    for (size_t segment_id = 0; segment_id < segment_history.size(); ++segment_id)
    {
        const auto &history = segment_history[segment_id];
        if (history.empty())
        {
            continue;
        }

        auto segment_data = std::max_element(history.begin(), history.end(), key_function);

        best_segments.push_back(*segment_data);
    }

    return best_segments;
}

std::vector<std::set<int>> Forest::get_segments()
{
    std::vector<std::set<int>> segments;
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        int root = find(i);
        if (root >= segments.size())
        {
            segments.resize(root + 1);
        }
        segments[root].insert(i);
    }
    return segments;
}

std::vector<cv::Point2i> Forest::get_bounding_box(int node_id) const
{
    const std::set<int> &segment = segments[find(node_id)];
    std::vector<cv::Point2i> bounding_box;
    bounding_box.reserve(segment.size());

    for (int node_id : segment)
    {
        bounding_box.emplace_back(node_id % width, node_id / width);
    }

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (const cv::Point2i &point : bounding_box)
    {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }

    std::vector<cv::Point2i> result;
    result.emplace_back(min_x, min_y);
    result.emplace_back(max_x, max_y);

    return result;
}

// Function to sort edges by weight
bool compareEdgesByWeight(const Edge &edge1, const Edge &edge2)
{
    return edge1.weight < edge2.weight;
}

std::pair<Forest, std::vector<Edge>> segment_graph(const cv::Mat &flow,
                                  const std::vector<Edge> &graph_edges, const cv::Mat &bev, const cv::Matx33f &persp_mat,
                                  const cv::Matx33f &inv_mat, const std::vector<cv::Matx33f> &inv_mat_upper)
{
    // Create a Forest object
    Forest forest(flow, bev, persp_mat, inv_mat, inv_mat_upper);

    // Define a lambda function to extract the weight of an edge
    auto weight = [](const Edge & edge) { return edge.weight; };

    // Sort the graph edges by weight
    std::vector<Edge> sorted_graph = graph_edges;
    std::sort(sorted_graph.begin(), sorted_graph.end(), compareEdgesByWeight);

    // Iterate through sorted edges and merge connected components in the forest
    for (const Edge &edge : sorted_graph)
    {
        int a = forest.find(edge.start);
        int b = forest.find(edge.end);

        if (a != b)
        {
            forest.new_merge(a, b);
        }
    }

    // Return the forest and sorted_graph as a pair
    return std::make_pair(forest, sorted_graph);
}

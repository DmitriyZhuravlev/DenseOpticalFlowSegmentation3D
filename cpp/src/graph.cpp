#include <spdlog/spdlog.h>
#include "graph.hpp"
//#include "lifting_3d.hpp"

#include <spdlog/spdlog.h>

extern std::shared_ptr<spdlog::logger> logger;

Solution get_bottom_variants(const cv::Point2f &orig_mov_dir,
                             const std::vector<cv::Point2i> &box_2d,
                             const cv::Matx33f &mat, const cv::Matx33f &inv_mat,
                             const cv::Matx33f &inv_matrix_upper, int cls);

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
SegmentData::SegmentData(double &score, std::set<int> seg, Solution &sol, double &move)
    : score(score), seg(seg), sol(sol), move(move)
{
    // Initialize member variables with provided values.
}

// Constructor definition
SegmentData::SegmentData()
    : score(-1.0)
{
    // Initialize member variables with provided values.
}

Node::Node(int id, const std::complex<float> &complex_value) : id(id), parent(id), rank(0), size(1),
    flow_value(complex_value.real(), complex_value.imag()) {}

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
        nodes.emplace_back(i, flow.at<std::complex<float>>(i / flow.cols, i % flow.cols));
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

int Forest::merge(int a, int b)
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
    return parent_b;
}

void Forest::LogInfo() const
{
    logger->info("Number of sets: {}", num_sets);
    logger->info("Width: {}", width);
    logger->info("Height: {}", height);
    logger->info("Minimum move: {}", min_move);
    logger->info("Segment history: {}", segment_history.size());
    // Iterate through the outer vector and print the sizes
    //for (size_t i = 0; i < segment_history.size(); ++i) {
    //std::cout << "Segment History " << i << " Size: " << segment_history[i].size() << std::endl;
    //}

    // Print bev, persp_mat, and inv_mat_upper as needed
    // For example, to print bev:
    // logger->info("bev:\n{}", bev);

    // Print inv_mat and inv_mat_upper as needed

    // Remember to include logger->info for each attribute you want to print
}

std::tuple<double, Solution> get_score(const std::vector<cv::Point2i> &bbox,
                                       const cv::Vec2f &direction,
                                       const cv::Mat &bev, const cv::Matx33f &persp_mat, const cv::Matx33f &inv_mat,
                                       const std::vector<cv::Matx33f> &inv_mat_upper)
{
    logger->info("{} ", __func__);

    double max_score = -1.0;
    //Solution(int cls, const std::vector<std::pair<double, double>> &ps_bev, const std::vector<std::pair<double, double>> &lower_face,
    //const std::vector<std::pair<double, double>> &upper_face,
    //const std::vector<std::pair<double, double>> &rectangle, double w_error, double h_error, double orient)
    Solution best_sol; //(0, cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), 0.0, 0.0, 0.0);
    //return std::make_tuple(max_score, best_sol);

    for (int cls = 0; cls < 3; ++cls)
    {

        Solution sol = get_bottom_variants(direction, bbox, persp_mat, inv_mat, inv_mat_upper[cls], cls);

        if (!sol.rectangle.empty() && max_score < (sol.w_error + sol.h_error) / 2)
        {
            logger->info("Rectangle with score: {} ", (sol.w_error + sol.h_error) / 2);
            max_score = (sol.w_error + sol.h_error) / 2;
            best_sol = sol;
        }
        else
        {
            logger->info("Not a rectangle");
        }
    }

    return std::make_tuple(max_score, best_sol);
}

void Forest::new_merge(int a, int b, double score_threshold, int min_size, double min_move,
                       double min_convexity)
{
    logger->info("{} ", __func__);
    int parent_b = merge(a, b);

    if (nodes[parent_b].size < min_size)
    {
        logger->warn("Low size: {} < {}", nodes[parent_b].size, min_size);
        return;
    }

    int x = parent_b % width;
    int y = parent_b / width;
    if (y < height / 10)
    {
        logger->warn("Low y {} < {}", y, height / 10);
        return;
    }

    double move = cv::norm(nodes[parent_b].flow_value);

    if (move < (y + 1) / static_cast<double>(height))
    {
        logger->warn("Low movement {} < {}", move, (y + 1) / static_cast<double>(height));
        return;
    }

    std::vector<cv::Point2i> bbox = get_bounding_box(parent_b);
    double rect_area = ((bbox[1].x - bbox[0].x + 1) * (bbox[1].y - bbox[0].y + 1));
    logger->info("Rect area: {}", rect_area);
    double convexity = nodes[parent_b].size / rect_area;
    logger->info("Node size: {}", nodes[parent_b].size);
    logger->info("Computed convexity: {}", convexity);
    //TODO add check min convexity threshold
    try
    {

        auto [score, solution] = get_score(bbox,
                                           nodes[parent_b].flow_value, bev,
                                           persp_mat,
                                           inv_mat,
                                           inv_mat_upper);

        if (score == -1)
        {
            logger->warn("Low score");
            return;
        };

        logger->info("Solution computed");

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
        logger->info("Computed min convexity: {}", min_convexity);

        if (convexity < min_convexity)
        {
            logger->warn("Low convexity");
            //return;
        }

        if (score > score_threshold)
        {
            std::set<int> seg = segments[parent_b];
            SegmentData data = segment_history[parent_b];
            if (data.score < score)
                segment_history[parent_b] = SegmentData(score, seg, solution, move);
            logger->info("Add segment with score: {} ", score);
        }
        else
        {
            logger->warn("Low score: {} < {}", score, score_threshold);
        }
    }
    catch (const cv::Exception &e)
    {
        // Handle the OpenCV exception here, e.g., print an error message
        //std::cerr << "OpenCV exception: " << e.what() << std::endl;
        logger->error("OpenCV exception: {} ", e.what());
    }
}

double Forest::get_segment_best_score(int node_id) const
{
    return segment_scores[node_id];
}

std::vector<SegmentData> Forest::GetBestSegments()
{
    logger->info("{}", __func__);
    //std::vector<SegmentData> best_segments;
    //auto key_function = [](const auto & item1, const auto & item2)
    //{
        //double score1 = item1.score;
        //double score2 = item2.score;
        //const auto &segment1 = item1.seg;
        //const auto &segment2 = item2.seg;

        //// Compare first by score and then by segment size if scores are equal
        //if (score1 != score2)
        //{
            //return score1 < score2;
        //}
        //else
        //{
            //return segment1.size() < segment2.size();
        //}
    //};
    //logger->info("segment history {}", segment_history.size());
    //for (size_t segment_id = 0; segment_id < segment_history.size(); ++segment_id)
    //{
        //const auto &history = segment_history[segment_id];
        //if (history.empty())
        //{
            //logger->warn("History for segment {} is empty", segment_id);
            //continue;
        //}

        //auto segment_data = std::max_element(history.begin(), history.end(), key_function);
        //logger->info("Add best segment for {}", segment_id);

        //best_segments.push_back(*segment_data);
    //}

    return segment_history;
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
    //TODO store segments in (x, y) format
    logger->info("{} ", __func__);
    const std::set<int> &segment = segments[find(node_id)];
    std::vector<cv::Point2i> bounding_box;
    bounding_box.reserve(segment.size());
    logger->info("segment size: {}", segment.size());

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (int node_id : segment)
    {
        cv::Point2i point(node_id % width, node_id / width);
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
    forest.LogInfo();

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

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

using namespace cv;
using namespace std;

bool debug = false;

class DisjointSetForest {
public:
    vector<int> parent;
    vector<int> rank;
    int num_sets;

    DisjointSetForest(int size) : num_sets(size) {
        parent.resize(size);
        rank.resize(size, 0);
        for (int i = 0; i < size; ++i) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (x != parent[x]) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void unionSets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            num_sets--;
        }
    }
};

vector<pair<Point2f, Point2f>> buildGraph(const Mat& smooth, int width, int height, const function<float(const Mat&, int, int, int, int)>& diffFunc, bool is8Connected) {
    vector<pair<Point2f, Point2f>> graphEdges;
    graphEdges.reserve(is8Connected ? (width - 1) * (height - 1) * 8 : (width - 1) * (height - 1) * 4);

    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            Point2f source(static_cast<float>(x), static_cast<float>(y));
            Point2f right(static_cast<float>(x + 1), static_cast<float>(y));
            Point2f down(static_cast<float>(x), static_cast<float>(y + 1));
            Point2f rightDown(static_cast<float>(x + 1), static_cast<float>(y + 1));

            float weight1 = diffFunc(smooth, x, y, x + 1, y);
            float weight2 = diffFunc(smooth, x, y, x, y + 1);
            graphEdges.emplace_back(source, right);
            graphEdges.emplace_back(source, down);
            if (is8Connected) {
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

vector<pair<Point2f, Point2f>> segmentGraph(const Mat& smooth, const vector<pair<Point2f, Point2f>>& graphEdges, int size, float k, int minCompSize, float threshold) {
    DisjointSetForest forest(size);
    vector<pair<Point2f, Point2f>> sortedGraph = graphEdges;
    vector<float> thresholdValues(size, 0.0f);

    sort(sortedGraph.begin(), sortedGraph.end(), [&](const pair<Point2f, Point2f>& a, const pair<Point2f, Point2f>& b) {
        float weightA = diff(smooth, static_cast<int>(a.first.x), static_cast<int>(a.first.y), static_cast<int>(a.second.x), static_cast<int>(a.second.y));
        float weightB = diff(smooth, static_cast<int>(b.first.x), static_cast<int>(b.first.y), static_cast<int>(b.second.x), static_cast<int>(b.second.y));
        return weightA < weightB;
    });

    for (size_t i = 0; i < sortedGraph.size(); ++i) {
        const pair<Point2f, Point2f>& edge = sortedGraph[i];
        int node1 = static_cast<int>(edge.first.y) * smooth.cols + static_cast<int>(edge.first.x);
        int node2 = static_cast<int>(edge.second.y) * smooth.cols + static_cast<int>(edge.second.x);
        int root1 = forest.find(node1);
        int root2 = forest.find(node2);

        if (root1 != root2) {
            if (diff(smooth, static_cast<int>(edge.first.x), static_cast<int>(edge.first.y), static_cast<int>(edge.second.x), static_cast<int>(edge.second.y)) < thresholdValues[root1] &&
                diff(smooth, static_cast<int>(edge.first.x), static_cast<int>(edge.first.y), static_cast<int>(edge.second.x), static_cast<int>(edge.second.y)) < thresholdValues[root2]) {
                forest.unionSets(node1, node2);
                root1 = forest.find(node1);
                if (thresholdValues[root2] < k) {
                    thresholdValues[root1] += thresholdValues[root2];
                } else {
                    thresholdValues[root1] += k;
                }
                thresholdValues[root2] = root1;
            }
        }
    }

    return sortedGraph;
}

float diff(const Mat& img, int x1, int y1, int x2, int y2) {
    float out = 0.0f;
    for (int c = 0; c < img.channels(); ++c) {
        out += static_cast<float>(pow(img.at<Vec3b>(y1, x1)[c] - img.at<Vec3b>(y2, x2)[c], 2));
    }
    return out;
}

float threshold(int size, float constant) {
    return constant / static_cast<float>(size);
}

Mat generateImage(const DisjointSetForest& forest, int width, int height, float threshold, const Mat& inputImage) {
    vector<Vec3b> colors(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int node = y * width + x;
            int root = forest.find(node);

            if (diff(inputImage, x, y, x, y) > threshold) {
                colors[root] = inputImage.at<Vec3b>(y, x);
            }
        }
    }

    Mat image(height, width, CV_8UC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int node = y * width + x;
            int root = forest.find(node);
            image.at<Vec3b>(y, x) = colors[root];
        }
    }

    return image;
}

int main() {
    // Load the consecutive images and convert them to grayscale
    Mat im1 = imread("/home/dzhura/ComputerVision/3dim-optical-flow/img/frame_1052.png");
    Mat im2 = imread("/home/dzhura/ComputerVision/3dim-optical-flow/img/frame_1053.png");
    Mat gray1, gray2;
    cvtColor(im1, gray1, COLOR_BGR2GRAY);
    cvtColor(im2, gray2, COLOR_BGR2GRAY);

    // Create a mask in HSV Color Space
    Mat hsv(im1.size(), CV_8UC3);
    // Set image saturation to maximum
    hsv.setTo(Scalar(0, 255, 255));

    Mat flow;
    calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    Mat magnitude, angle;
    cartToPolar(flow[..., 0], flow[..., 1], magnitude, angle);

    // Set image hue according to the optical flow direction
    angle.convertTo(hsv[..., 0], CV_8U, 0.5, 0.5);
    // Set image value according to the optical flow magnitude (normalized)
    normalize(magnitude, hsv[..., 2], 0, 255, NORM_MINMAX);

    // Define arrow length and scale factor for visualization
    int arrowLength = 10;
    int arrowScale = 20;

    Mat arrowHsv = hsv.clone();

    for (int y = 0; y < flow.rows; y += arrowScale) {
        for (int x = 0; x < flow.cols; x += arrowScale) {
            int dx = static_cast<int>(flow.at<Vec2f>(y, x)[0] * arrowLength);
            int dy = static_cast<int>(flow.at<Vec2f>(y, x)[1] * arrowLength);
            Point start(x, y);
            Point end(x + dx, y + dy);
            arrowedLine(arrowHsv, start, end, Scalar(58, 50, 100), 1, 8, 0, 0.2);
        }
    }

    // Convert HSV to RGB (BGR) color representation
    Mat rgb;
    cvtColor(arrowHsv, rgb, COLOR_HSV2BGR);

    // Display the Output
    imwrite("output/Dense Output.jpg", rgb);
    imshow("RGB Image", rgb);
    waitKey(0);

    return 0;
}

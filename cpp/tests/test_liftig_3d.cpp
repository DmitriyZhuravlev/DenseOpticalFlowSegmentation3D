#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "lifting_3d.hpp" // Include the header file containing your functions

std::shared_ptr<spdlog::logger> logger;

void init_logger() {
    // Create a logger with the name "my_logger" and use the stdout color sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    logger = std::make_shared<spdlog::logger>("seg", console_sink);
    
    // Set the log level (optional)
    logger->set_level(spdlog::level::info); // You can change the log level as needed
    
    // Register the logger globally
    spdlog::register_logger(logger);
}

// Define a fixture class for your tests
class Lifting3DTest : public ::testing::Test {
protected:
    virtual void SetUp() {

        // Setup code, if needed
    }

    virtual void TearDown() {
        // Teardown code, if needed
    }
};

// Define test cases for your functions
TEST_F(Lifting3DTest, TestGetUpperFaceSimple) {
    // Define input box_2d and lower_face vectors
    std::vector<cv::Point2i> box_2d = {cv::Point2i(0, 0), cv::Point2i(100, 100)};
    std::vector<cv::Point2f> lower_face = {cv::Point2f(0.0, 0.0), cv::Point2f(100.0, 0.0),
                                           cv::Point2f(100.0, 50.0), cv::Point2f(0.0, 50.0)};

    // Calculate the expected result
    std::vector<cv::Point2f> expected_result = {cv::Point2f(0.0, -50.0), cv::Point2f(100.0, -50.0),
                                                cv::Point2f(100.0, 0.0), cv::Point2f(0.0, 0.0)};

    // Call the function
    std::vector<cv::Point2f> result = get_upper_face_simple(box_2d, lower_face);

    // Check if the result matches the expected result
    EXPECT_EQ(result, expected_result);
}

TEST_F(Lifting3DTest, TestGetUpperFace) {
    // Define input box_2d and lower_face vectors
    std::vector<cv::Point2i> box_2d = {cv::Point2i(0, 0), cv::Point2i(100, 100)};
    std::vector<cv::Point2f> lower_face = {cv::Point2f(0.0, 0.0), cv::Point2f(100.0, 0.0),
                                           cv::Point2f(100.0, 50.0), cv::Point2f(0.0, 50.0)};

    // Calculate the expected result
    std::vector<cv::Point2f> expected_result = {cv::Point2f(0.0, 0.0), cv::Point2f(0.0, -50.0),
                                                cv::Point2f(100.0, -50.0), cv::Point2f(100.0, 0.0)};

    // Call the function
    std::vector<cv::Point2f> result = get_upper_face(box_2d, lower_face);

    // Check if the result matches the expected result
    EXPECT_EQ(result, expected_result);
}

TEST_F(Lifting3DTest, TestGetObjSize) {
    // Test for class 0
    std::pair<double, double> result0 = get_obj_size(0);
    EXPECT_DOUBLE_EQ(result0.first, 258.0);
    EXPECT_DOUBLE_EQ(result0.second, 84.0);

    // Test for class 1
    std::pair<double, double> result1 = get_obj_size(1);
    EXPECT_DOUBLE_EQ(result1.first, 349.0);
    EXPECT_DOUBLE_EQ(result1.second, 165.0);

    // Test for class 2
    std::pair<double, double> result2 = get_obj_size(2);
    EXPECT_DOUBLE_EQ(result2.first, 370.0);
    EXPECT_DOUBLE_EQ(result2.second, 180.0);
}

// Add more test cases for other functions

int main(int argc, char** argv) {
    init_logger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

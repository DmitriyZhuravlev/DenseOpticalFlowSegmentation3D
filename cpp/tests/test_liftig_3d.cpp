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

//cv::Point2f get_intersect(cv::Point2f a1, cv::Point2f a2, cv::Point2f b1, cv::Point2f b2) {
    //float a1x = a2.x - a1.x;
    //float a1y = a2.y - a1.y;
    //float b1x = b2.x - b1.x;
    //float b1y = b2.y - b1.y;

    //float det = a1x * b1y - a1y * b1x;

    //if (std::abs(det) < 1e-9) {
        //// Lines are parallel, no intersection
        //return cv::Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    //} else {
        //float c1 = a1x * a1.y - a1y * a1.x;
        //float c2 = b1x * b1.y - b1y * b1.x;

        //float x = (c1 * b1x - c2 * a1x) / det;
        //float y = (c1 * b1y - c2 * a1y) / det;

        //return cv::Point2f(x, y);
    //}
//}

// Test fixture for the intersection function
class IntersectionTest : public ::testing::Test {
protected:
    cv::Point2f A1, A2, B1, B2;

    void SetUp() override {
        A1 = cv::Point2f(1.0, 1.0);
        A2 = cv::Point2f(5.0, 5.0);
        B1 = cv::Point2f(1.0, 5.0);
        B2 = cv::Point2f(5.0, 1.0);
    }
};

TEST_F(IntersectionTest, IntersectionExists) {
    cv::Point2f A1(1.0, 1.0);
    cv::Point2f A2(4.0, 4.0);
    cv::Point2f B1(1.0, 8.0);
    cv::Point2f B2(2.0, 4.0);

    cv::Point2f intersection = get_intersect(A1, A2, B1, B2);
    ASSERT_NEAR(2.4f, intersection.x, 1e-2);
    ASSERT_NEAR(2.4f, intersection.x, 1e-2); 
}

TEST_F(IntersectionTest, IntersectionDoesNotExist) {
    cv::Point2f A1(1.0, 1.0);
    cv::Point2f A2(1.0, 2.0);
    cv::Point2f B1(3.0, 3.0);
    cv::Point2f B2(3.0, 4.0);

    cv::Point2f intersection = get_intersect(A1, A2, B1, B2);
    EXPECT_TRUE(cvIsNaN(intersection.x));
    EXPECT_TRUE(cvIsNaN(intersection.y));
}

//// Define test cases for your functions
//TEST_F(Lifting3DTest, TestGetUpperFaceSimple) {
    //// Define input box_2d and lower_face vectors
    //std::vector<cv::Point2i> box_2d = {cv::Point2i(0, 0), cv::Point2i(100, 100)};
    //std::vector<cv::Point2f> lower_face = {cv::Point2f(0.0, 0.0), cv::Point2f(100.0, 0.0),
                                           //cv::Point2f(100.0, 50.0), cv::Point2f(0.0, 50.0)};

    //// Calculate the expected result
    //std::vector<cv::Point2f> expected_result = {cv::Point2f(0.0, -50.0), cv::Point2f(100.0, -50.0),
                                                //cv::Point2f(100.0, 0.0), cv::Point2f(0.0, 0.0)};

    //// Call the function
    //std::vector<cv::Point2f> result = get_upper_face_simple(box_2d, lower_face);

    //// Check if the result matches the expected result
    //EXPECT_EQ(result, expected_result);
//}

//TEST_F(Lifting3DTest, TestGetUpperFace) {
    //// Define input box_2d and lower_face vectors
    //std::vector<cv::Point2i> box_2d = {cv::Point2i(0, 0), cv::Point2i(100, 100)};
    //std::vector<cv::Point2f> lower_face = {cv::Point2f(0.0, 0.0), cv::Point2f(100.0, 0.0),
                                           //cv::Point2f(100.0, 50.0), cv::Point2f(0.0, 50.0)};

    //// Calculate the expected result
    //std::vector<cv::Point2f> expected_result = {cv::Point2f(0.0, 0.0), cv::Point2f(0.0, -50.0),
                                                //cv::Point2f(100.0, -50.0), cv::Point2f(100.0, 0.0)};

    //// Call the function
    //std::vector<cv::Point2f> result = get_upper_face(box_2d, lower_face);

    //// Check if the result matches the expected result
    //EXPECT_EQ(result, expected_result);
//}

//TEST_F(Lifting3DTest, TestGetObjSize) {
    //// Test for class 0
    //std::pair<double, double> result0 = get_obj_size(0);
    //EXPECT_DOUBLE_EQ(result0.first, 258.0);
    //EXPECT_DOUBLE_EQ(result0.second, 84.0);

    //// Test for class 1
    //std::pair<double, double> result1 = get_obj_size(1);
    //EXPECT_DOUBLE_EQ(result1.first, 349.0);
    //EXPECT_DOUBLE_EQ(result1.second, 165.0);

    //// Test for class 2
    //std::pair<double, double> result2 = get_obj_size(2);
    //EXPECT_DOUBLE_EQ(result2.first, 370.0);
    //EXPECT_DOUBLE_EQ(result2.second, 180.0);
//}


//input
//orig_mov_dir: [2.5470946 1.9316475]
//box_2d: (375, 92, 576, 286)
//mat: [[ 2.01377838e+01 -1.34744920e+01  4.02174272e+02]
 //[ 5.11635077e+00  8.00335022e+02 -6.22513321e+04]
 //[ 3.93565444e-04  3.97205947e-02  1.00000000e+00]]
//inv_mat: [[ 2.02212552e-01  1.81942728e-03  3.19370859e+01]
 //[-1.82975914e-03  1.23437589e-03  7.75774258e+01]
 //[-6.90475148e-06 -4.97462083e-05  1.00000000e+00]]
//inv_matrix_upper: [[ 2.03701900e-01  1.69508037e-03  3.23672674e+01]
 //[ 0.00000000e+00  1.46371164e-03  2.96614710e+01]
 //[-0.00000000e+00 -5.01704822e-05  1.00000000e+00]]
//cls: 2
//output
//ps_bev [(327.809749190909, 13476.772230116465), (1398.2414179174136, 2769.3562851313454), (2204.8576245955073, 2935.1653236816246), (647.3324083001849, 13473.77576520306)]
//lower_face [[385.305   286.     ]
 //[375.      269.47327]
 //[555.45557 270.2111 ]
 //[576.      286.92706]]
//upper_face [[385.305    99.43571]
 //[375.       92.75792]
 //[555.45557  92.     ]
 //[576.       98.69487]]
//ps_bev [(327.809749190909, 13476.772230116465), (1398.2414179174136, 2769.3562851313454), (2204.8576245955073, 2935.1653236816246), (647.3324083001849, 13473.77576520306)]
//ps_bev [(327.809749190909, 13476.772230116465), (1398.2414179174136, 2769.3562851313454), (2204.8576245955073, 2935.1653236816246), (647.3324083001849, 13473.77576520306)]
//mov_angle -1.6261444189491607
//corners [(344.2895075890286, 13476.617683900684), (364.7578474586475, 13107.18427033616), (664.9228681919458, 13123.814817102848), (644.4545283223268, 13493.248230667372)]

//[2023-10-11 23:03:47.865] [seg] [info] corners :
//[2023-10-11 23:03:47.865] [seg] [info] 471.06122 13475.429
//[2023-10-11 23:03:47.865] [seg] [info] 295.38992 13801.066
//[2023-10-11 23:03:47.865] [seg] [info] 458.42413 13888.9795
//[2023-10-11 23:03:47.865] [seg] [info] 634.0954 13563.342


TEST(GetBottomVariantsTest, Test1) {

    cv::Point2f orig_mov_dir(2.5470946f, 1.9316475f);
    std::vector<cv::Point2i> box_2d = {cv::Point2i(375, 92), cv::Point2i(576, 286)};
    cv::Matx33f mat(20.1377838f, -13.4744920f, 402.174272f, 5.11635077f, 800.335022f, -62251.3321f, 0.000393565444f, 0.0397205947f, 1.0f);
    cv::Matx33f inv_mat(0.202212552f, 0.00181942728f, 31.9370859f, -0.00182975914f, 0.00123437589f, 77.5774258f, -6.90475148e-06f, -4.97462083e-05f, 1.0f);
    cv::Matx33f inv_matrix_upper(0.203701900f, 0.00169508037f, 32.3672674f, 0.0f, 0.00146371164f, 29.6614710f, 0.0f, -5.01704822e-05f, 1.0f);
    int cls = 2;

    Solution expected;
    expected.cls = 2;
    expected.ps_bev = {{327.809749190909f, 13476.772230116465f},
                       {1398.2414179174136f, 2769.3562851313454f},
                       {2204.8576245955073f, 2935.1653236816246f},
                       {647.3324083001849f, 13473.77576520306f}};
    expected.lower_face = {{385.305f, 286.0f},
                           {375.0f, 269.47327f},
                           {555.45557f, 270.2111f},
                           {576.0f, 286.92706f}};
    expected.upper_face = {{385.305f, 99.43571f},
                           {375.0f, 92.75792f},
                           {555.45557f, 92.0f},
                           {576.0f, 98.69487f}};
    expected.rectangle = {};  // Fill in with appropriate values
    expected.w_error = 0.5987518562843858;
    expected.h_error = 0.7156805292391223;
    expected.orient = -1.6261444189491607;

    // Call the get_bottom_variants function and store the result in a Solution object
    Solution actual = get_bottom_variants(orig_mov_dir, box_2d, mat, inv_mat, inv_matrix_upper, cls);


    // Add your assertions here
    //ASSERT_EQ(solution.ps_bev, actual.ps_bev);
    for (size_t i = 0; i < expected.ps_bev.size(); ++i) {
    ASSERT_NEAR(expected.ps_bev[i].x, actual.ps_bev[i].x, 1e-1); // Tolerance of 0.01 for x-coordinate
    ASSERT_NEAR(expected.ps_bev[i].y, actual.ps_bev[i].y, 1e-1); // Tolerance of 0.01 for y-coordinate
    
    ASSERT_NEAR(expected.lower_face[i].x, actual.lower_face[i].x, 1e-1); // Tolerance of 0.01 for x-coordinate
    ASSERT_NEAR(expected.lower_face[i].y, actual.lower_face[i].y, 1e-1); // Tolerance of 0.01 for y-coordinate
    
    ASSERT_NEAR(expected.upper_face[i].x, actual.upper_face[i].x, 1e-1); // Tolerance of 0.01 for x-coordinate
    ASSERT_NEAR(expected.upper_face[i].y, actual.upper_face[i].y, 1e-1); // Tolerance of 0.01 for y-coordinate
}

    ASSERT_NEAR(expected.w_error, actual.w_error, 1e-1);
    ASSERT_NEAR(expected.h_error, actual.h_error, 1e-1);
    ASSERT_NEAR(expected.orient, actual.orient, 1e-1);
}

int main(int argc, char** argv) {
    init_logger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

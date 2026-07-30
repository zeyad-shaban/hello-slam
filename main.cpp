#include <Eigen/Core>
#include <Eigen/Eigen>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <ratio>

int main() {
  auto left_img = cv::imread("C:/Users/zeyad/Desktop/2_Projects/1Active/"
                             "slam_book_follow/data/left.png",
                             0);
  auto right_img = cv::imread("C:/Users/zeyad/Desktop/2_Projects/1Active/"
                              "slam_book_follow/data/right.png",
                              0);
  if (left_img.empty() || right_img.empty()) {
    std::cout << "either left_img {" << left_img.empty() << "} or right_img {"
              << right_img.empty() << "} was not found" << std::endl;
    return 1;
  }
  std::cout << left_img.rows << " x " << left_img.cols << " x "
            << left_img.channels() << std::endl;
  std::cout << right_img.rows << " x " << right_img.cols << " x "
            << right_img.channels() << std::endl;

  double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
  double b = 0.573;

  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
      0, 150, 5, 8 * 1 * 5 * 5, 32 * 1 * 5 * 5, 1, 61, 10, 100, 32);
  cv::Mat disparity_sgbm, disparity;
  sgbm->compute(left_img, right_img, disparity_sgbm);
  disparity_sgbm.convertTo(disparity, CV_32FC1, 1.0f / 16.0f);

  std::vector<Eigen::Vector4d> point_cloud;
  point_cloud.reserve(left_img.cols * left_img.rows);

  auto t1 = std::chrono::steady_clock::now(); // this loop takes 4ms to complete, 250 fps a python developer would never dream off.. ehm sorry
  for (int v = 0; v < left_img.rows; v++) {
    auto disparity_row = disparity.ptr<float>(v);
    for (int u = 0; u < left_img.cols; u++) {
      double d = disparity_row[u];
      if (d <= 0)
        continue;

      double Z = (fx * b) / d;

      double x = (u - cx) / fx; // u = fx x + cx
      double y = (v - cy) / fy;

      double X = Z * x;
      double Y = Z * y;
      point_cloud.push_back(
          Eigen::Vector4d(X, Y, Z, left_img.ptr<uchar>(v)[u]));
    }
  }
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "time took: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count()
            << std::endl;

  // cv::imshow("disparity", disparity / 32);
  // cv::waitKey(0);

  // std::ofstream out("_pointcloud.ply");

  // // 1. Write the PLY file header (Geometry + Blue Color)
  // out << "ply\n";
  // out << "format ascii 1.0\n";
  // out << "element vertex " << point_cloud.size() << "\n";
  // out << "property float x\n";
  // out << "property float y\n";
  // out << "property float z\n";
  // out << "property uchar red\n";
  // out << "property uchar green\n";
  // out << "property uchar blue\n";
  // out << "end_header\n";

  // // 2. Write the X, Y, Z coordinates and RGB values
  // for (const auto &p : point_cloud) {
  //   int color = static_cast<int>(p[3]);
  //   if (p[2] >= 50)
  //     continue;
  //   out << p[0] << " " << p[1] << " " << p[2] << " " << color << " " << color
  //   << " " << color << "\n";
  // };

  // out.close();
  // std::cout << "Done exporting point cloud" << std::endl;

  return 0;
}
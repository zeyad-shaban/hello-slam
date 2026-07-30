#include <cmath>
#include <iostream>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int main() {
  cv::Mat img = cv::imread("C:/Users/zeyad/Desktop/2_Projects/1Active/"
                           "slam_book_follow/data/distorted.png");
  if (img.empty()) {
    std::cout << "image is empty, perhaps invalid path" << std::endl;
    return 1;
  }

  std::cout << "image shape: " << img.rows << ", " << img.cols << ", "
            << img.channels() << std::endl;

  cv::Mat img_undist(img.rows, img.cols, img.type());
  
  std::cout << "undistorted img shape: " << img_undist.rows << ", " << img_undist.cols << ", "
            << img_undist.channels() << std::endl;

  // rad−tan model params
  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359,
         p2 = 1.76187114e-05;
  double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

  // Get location of distortion
  for (int v = 0; v < img.rows; ++v) {
    auto undist_row_ptr = img_undist.ptr<cv::Vec3b>(v);

    for (int u = 0; u < img.cols; ++u) {
      double x = (u - cx) / fx;
      double y = (v - cy) / fy;

      double r = std::sqrt(x * x + y * y);
      double radial_err_scale = 1 + k1 * std::pow(r, 2) + k2 * std::pow(r, 4);

      double x_dist =
          x * (radial_err_scale) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
      double y_dist =
          y * (radial_err_scale) + 2 * p2 * x * y + p1 * (r * r + 2 * y * y);

      double u_dist = fx * x_dist + cx;
      double v_dist = fy * y_dist + cy;

      // Bilinear Interpolation
      int u_int = std::floor(u_dist);
      int v_int = std::floor(v_dist);
      double du = u_dist - u_int;
      double dv = v_dist - v_int;

      if (u_int < 0 || u_int >= img.cols || v_int < 0 || v_int >= img.rows) {
        std::cout << "out of range: u dist: " << u_dist << " v_dist: " << v_dist
                  << "\n";
        undist_row_ptr[u] = cv::Vec3b(0, 0, 0);
        continue;
      }

      int u_next = std::min(u_int + 1, img_undist.cols - 1);
      int v_next = std::min(v_int + 1, img_undist.rows - 1);
      
      if (u_next != u_int + 1 || v_next != v_int + 1) {
        std::cout << "bruh" << std::endl;
      }

      auto p00 = img.ptr<cv::Vec3b>(v_int)[u_next];
      auto p01 = img.ptr<cv::Vec3b>(v_int)[u_next];
      auto p10 = img.ptr<cv::Vec3b>(v_next)[u_int];
      auto p11 = img.ptr<cv::Vec3b>(v_next)[u_next];

      undist_row_ptr[u] = p00 * (1 - du) * (1 - dv) + p01 * du * (1 - dv) +
                          p10 * (1 - du) * dv + p11 * du * dv;
    }
  }

  cv::imshow("distorted", img);
  cv::imshow("undistorted", img_undist);

  cv::waitKey(0);

  return 0;
}
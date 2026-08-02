#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <ratio>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <sstream>
#include <string>

int main() {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
  const int N_IMAGES = 5;

  const std::string data_path =
      "C:/Users/zeyad/Desktop/2_Projects/1Active/slam_book_follow/data";

  std::ifstream pose_stream(data_path + "/rgbd_pose.txt");
  if (!pose_stream.is_open()) {
    std::cerr << "no pose file found" << std::endl;
    return 1;
  }

  std::string line;
  std::getline(pose_stream, line);

  // performance_try: count lines first and reserve required amount in advanced
  // performance_try: dont' allocate quaternion and vector3d t, just put them
  // directly performance_try: use push_back instead of embrace_back

  auto t1 = std::chrono::steady_clock::now();
  std::vector<Sophus::SE3d> pose_vec;
  std::array<double, 7> temp_pose_vec;

  std::getline(pose_stream, line);
  while (std::getline(pose_stream, line)) {
    if (line.empty())
      continue;
    const char *ptr = line.data();
    const char *end = line.data() + line.size();

    for (int i = 0; i < 7; ++i) {
      while (ptr < end && (*ptr == ' ' || *ptr == ','))
        ++ptr;

      auto [next_ptr, ec] = std::from_chars(ptr, end, temp_pose_vec[i]);

      if (ec != std::errc{}) {
        std::cerr << "error occured while parsing file.."
                  << std::make_error_code(ec).message() << std::endl;
        return 1;
      }
      ptr = next_ptr;
    }

    const Eigen::Quaternion q(temp_pose_vec[6], temp_pose_vec[3],
                              temp_pose_vec[4], temp_pose_vec[5]);
    const Eigen::Vector3d t(temp_pose_vec[0], temp_pose_vec[1],
                            temp_pose_vec[2]);
    pose_vec.emplace_back(q, t);
  }
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "1. time took to create the pose: "
            << std::chrono::duration<double, std::milli>(t2 - t1) << std::endl;

  // ====reading the images====
  std::vector<cv::Mat> rgb_images;
  std::vector<cv::Mat> depth_images;

  for (size_t i = 1; i < N_IMAGES + 1; ++i) {
    rgb_images.push_back(
        cv::imread(data_path + "/rgbd_color/" + std::to_string(i) + ".png",
                   cv::IMREAD_COLOR));

    depth_images.push_back(
        cv::imread(data_path + "/rgbd_depth/" + std::to_string(i) + ".pgm",
                   cv::IMREAD_UNCHANGED));
  }

  double cx = 325.5;
  double cy = 253.5;
  double fx = 518.0;
  double fy = 519.0;

  t1 = std::chrono::steady_clock::now();
  std::vector<Eigen::Matrix<double, 6, 1>> pointcloud;
  pointcloud.reserve(rgb_images[0].cols * rgb_images[0].rows * N_IMAGES);

  for (size_t i = 0; i < N_IMAGES; ++i) {
    const auto depth_img = depth_images[i];
    const auto rgb_img = rgb_images[i];

    for (size_t v = 0; v < rgb_img.rows; ++v) {
      const cv::Vec3b *row_ptr = rgb_img.ptr<cv::Vec3b>(v);
      const ushort *depth_row = depth_img.ptr<ushort>(v);

      for (size_t u = 0; u < rgb_img.cols; ++u) {
        double Z_cam = depth_row[u] / 1000.0f; // mm
        if (Z_cam <= 0)
          continue;

        double X_cam = Z_cam * (u - cx) / fx;
        double Y_cam = Z_cam * (v - cy) / fy;
        Eigen::Vector3d p_cam(X_cam, Y_cam, Z_cam);
        // Eigen::Vector3d p_world = pose_vec[i] * p_cam;
        Eigen::Vector3d p_world = p_cam;

        pointcloud.emplace_back(p_world[0], p_world[1], p_world[2],
                                row_ptr[u][0], row_ptr[u][1], row_ptr[u][2]);
      }
    }
  }

  // Save PLY format
  t1 = std::chrono::steady_clock::now();
  std::cout << "Starting exporting PLY format..." << std::endl;

  std::ofstream out(data_path + "/created/pointcloud_rgbd.ply",
                    std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "failed to open .ply file to write pointcloud" << std::endl;
    return 1;
  }

  out << "ply\n";
  out << "format binary_little_endian 1.0\n";
  out << "element vertex " << pointcloud.size() << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property uchar red\n";
  out << "property uchar green\n";
  out << "property uchar blue\n";
  out << "end_header\n";

  for (const auto &point : pointcloud) {
    // 1. Cast the 64-bit doubles down to 32-bit floats
    float xyz[3] = {static_cast<float>(point[0]), static_cast<float>(point[1]),
                    static_cast<float>(point[2])};

    // 2. Cast the colors to 8-bit unsigned chars, reordering BGR to RGB
    uint8_t rgb[3] = {static_cast<uint8_t>(point[5]),
                      static_cast<uint8_t>(point[4]),
                      static_cast<uint8_t>(point[3])};

    // 3. Dump the raw memory bytes straight to the file buffer
    out.write(reinterpret_cast<const char *>(xyz), sizeof(xyz));
    out.write(reinterpret_cast<const char *>(rgb), sizeof(rgb));
  }

  t2 = std::chrono::steady_clock::now();
  std::cout << "done saving it succesfully... took: " << std::chrono::duration<double, std::milli>(t2 - t1)  << std::endl;

  return 0;
}
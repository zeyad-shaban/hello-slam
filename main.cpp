#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <sstream>
#include <string>
#include <vector>

int main() {
  const int N_COLS = 15;
  std::ifstream fin("C:/Users/zeyad/Desktop/2_Projects/1Active/"
                    "slam_book_follow/data/slam_trajectory_data.csv");

  std::string line;
  std::vector<double> row(N_COLS);
  // time,gt_tx,gt_ty,gt_tz,gt_qx,gt_qy,gt_qz,gt_qw,est_tx,est_ty,est_tz,est_qx,est_qy,est_qz,est_qw
  //   0     1    2     3      4     5    6     7      8     9      10     11 12
  //   13     14
  //  13     14

  if (!fin) {
    std::cerr << "Failed to open file" << std::endl;
    return 1;
  }
  std::getline(fin, line);

  double ate_all{0};
  double ate_trans{0};
  double rpe_all{0};
  double rpe_trans{0};

  int i = 0;
  Sophus::SE3d T_gt_prev;
  Sophus::SE3d T_est_prev;

  while (std::getline(fin, line)) {
    std::stringstream line_streamed{line};
    std::string token;

    int j = 0;
    while (std::getline(line_streamed, token, ','))
      row[j++] = std::stod(token);

    Eigen::Quaterniond q_gt(row[7], row[4], row[5], row[6]);
    Eigen::Vector3d t_gt(row[1], row[2], row[3]);

    Eigen::Quaterniond q_est(row[14], row[11], row[12], row[13]);
    Eigen::Vector3d t_est(row[8], row[9], row[10]);

    Sophus::SE3d T_gt(q_gt, t_gt);
    Sophus::SE3d T_est(q_est, t_est);

    ate_all += std::pow((T_gt.inverse() * T_est).log().norm(), 2);
    ate_trans += std::pow((T_gt.inverse() * T_est).translation().norm(), 2);

    if (i != 0) {
      rpe_all += std::pow(((T_gt_prev.inverse() * T_gt).inverse() *
                           (T_est_prev.inverse() * T_est))
                              .log()
                              .norm(),
                          2);

      rpe_trans += std::pow(((T_gt_prev.inverse() * T_gt).inverse() *
                             (T_est_prev.inverse() * T_est))
                                .translation()
                                .norm(),
                            2);
    }

    T_gt_prev = T_gt;
    T_est_prev = T_est;

    i++;
  }

  ate_all = std::sqrt((1.0 / i) * ate_all);
  ate_trans = std::sqrt((1.0 / i) * ate_trans);
  rpe_all = std::sqrt((1.0 / (i - 1)) * rpe_all);
  rpe_trans = std::sqrt((1.0 / (i - 1)) * rpe_trans);

  std::cout << "ATE All (Absolute Trajectory Error All):           " << ate_all
            << "\n";
  std::cout << "ATE Trans (Absolute Trajectory Error Translation): "
            << ate_trans << "\n";
  std::cout << "RPE All (Relative Pose Error All):                 " << rpe_all
            << "\n";
  std::cout << "RPE Trans (Relative Pose Error Translation):       "
            << rpe_trans << "\n";

  return 0;
}
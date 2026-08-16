#include <Eigen/Dense>
#include <array>
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/problem.h>
#include <ceres/types.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <ratio>

double fx(const double a, const double b, const double c, const double x) {
  return exp(a * x * x + b * x + c);
}

struct CurveFittingCost {
private:
  const double _x;
  const double _y;

public:
  CurveFittingCost(double x, double y) : _x{x}, _y{y} {}

  template <typename T>

  bool operator()(const T *const abc, T *residual) const {
    T x_val = T(_x);
    T y_val = T(_y);

    residual[0] =
        y_val - ceres::exp(abc[0] * x_val * x_val + abc[1] * x_val + abc[2]);

    return true;
  }
};

int main() {
  static std::random_device rd;
  static std::mt19937 gen;
  static std::uniform_real_distribution<double> dist(-2, 2);

  const double a_gt = 0.1, b_gt = 0.05, c_gt = 2;
  double a_pred = -0.1, b_pred = -0.1, c_pred = -0.1;

  const int N_DATA = 1000;
  const int N_ITER = 100;

  std::array<double, N_DATA> x_arr;
  std::array<double, N_DATA> y_arr;

  for (size_t i = 0; i < N_DATA; ++i) {
    double x{i / 100.0f};
    double noise = dist(gen);
    double y = fx(a_gt, b_gt, c_gt, x) + noise;

    x_arr[i] = x;
    y_arr[i] = y;
  }
  
  std::cout << "biggest y value: " << y_arr[N_DATA - 1] << std::endl;

  auto t0 = std::chrono::steady_clock::now();
  double avg_errs = 0;
  for (size_t i = 0; i < N_ITER; ++i) {
    Eigen::Matrix<double, 3, 3> H = Eigen::Matrix<double, 3, 3>::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();

    avg_errs = 0;
    for (size_t j = 0; j < N_DATA; ++j) {
      const double x = x_arr[j];
      const double y = y_arr[j];
      const double y_pred = fx(a_pred, b_pred, c_pred, x);

      const double err = y - y_pred;
      const double derr_a = -x * x * exp(a_pred * x * x + b_pred * x + c_pred);
      const double derr_b = -x * exp(a_pred * x * x + b_pred * x + c_pred);
      const double derr_c = -exp(a_pred * x * x + b_pred * x + c_pred);

      // Hessian Matrix
      Eigen::Matrix<double, 1, 3> J_i{derr_a, derr_b, derr_c}; // 1x3
      H += J_i.transpose() * J_i;

      // gradient vector
      g += -J_i.transpose() * err;

      avg_errs += abs(err);
    }
    avg_errs /= N_DATA;

    Eigen::Vector3d dx = H.ldlt().solve(g);
    a_pred += dx(0);
    b_pred += dx(1);
    c_pred += dx(2);

    // std::cout << "a_pred: " << a_pred << ", b_pred: " << b_pred << ", c_pred:
    // " << c_pred << ", avg_errs: " << avg_errs << std::endl;

    if (dx.norm() <= 0.0001) {
      std::cout << "early break" << std::endl;
      break;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

  std::cout << "a_pred: " << a_pred << ", b_pred: " << b_pred
            << ", c_pred: " << c_pred << ", avg_errs: " << avg_errs
            << std::endl;
  std::cout << "Time taken: " << duration.count() << "ms" << std::endl;

  std::cout << "=====GOOGLE CERES=====" << std::endl;
  ceres::Problem problem;
  double abc[3] = {0, 0, 0};
  
  for (int i = 0; i < N_DATA; ++i) {
    auto* cost_function = new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(new CurveFittingCost(x_arr[i], y_arr[i]));
    
    problem.AddResidualBlock(cost_function, nullptr, abc);
  }
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  // options.minimizer_progress_to_stdout = false;
  
  t0 = std::chrono::steady_clock::now();
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  t1 = std::chrono::steady_clock::now();
  
  auto google_duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
  
  std::cout << "Google took: " << google_duration << std::endl;
  
  
  // std::cout << summary.BriefReport() << std::endl;
  std::cout << "a: " << abc[0] << ", b: " << abc[1] << ", c: " << abc[2] << std::endl;

  return 0;
}
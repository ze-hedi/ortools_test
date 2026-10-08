#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/lp_data/mps_reader.h"
#include "ortools/math_opt/cpp/math_opt.h"
#include "ortools/math_opt/io/mps_converter.h"

namespace math_opt = operations_research::math_opt;
using operations_research::MPSolver;
using operations_research::MPModelProto;

constexpr int kRuns = 10;
const std::string kMpsFile =
    "/home/bouchehdahed/studies/clean_big_study/15-15_1000/sub/sub_15.mps";

int main() {
  // --- Load via MathOpt and print dimensions ---
  auto model_proto_or = math_opt::ReadMpsFile(kMpsFile);
  if (!model_proto_or.ok()) {
    std::cerr << "Failed to read MPS (MathOpt): " << model_proto_or.status() << std::endl;
    return 1;
  }
  auto model_or = math_opt::Model::FromModelProto(*model_proto_or);
  if (!model_or.ok()) {
    std::cerr << "Failed to build model: " << model_or.status() << std::endl;
    return 1;
  }
  auto& model_ptr = *model_or;
  std::cout << "Model: " << kMpsFile << std::endl;
  std::cout << "Variables (cols): " << model_ptr->num_variables() << std::endl;
  std::cout << "Constraints (rows): " << model_ptr->num_linear_constraints() << std::endl;

  // --- Load via MPSolver ---
  auto mp_proto_or = operations_research::glop::MpsFileToMPModelProto(kMpsFile);
  if (!mp_proto_or.ok()) {
    std::cerr << "Failed to read MPS (MPSolver): " << mp_proto_or.status() << std::endl;
    return 1;
  }
  const MPModelProto& mp_proto = *mp_proto_or;

  std::vector<long long> mathopt_times, mpsolver_times;

  // --- MathOpt x10 ---
  for (int i = 0; i < kRuns; ++i) {
    // Rebuild model each time from proto
    auto m = math_opt::Model::FromModelProto(*model_proto_or);
    auto start = std::chrono::high_resolution_clock::now();
    const auto result = math_opt::Solve(**m, math_opt::SolverType::kGlop);
    auto end = std::chrono::high_resolution_clock::now();
    mathopt_times.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    if (i == 0) {
      if (result.ok())
        std::cout << "\nMathOpt optimal value: " << result->objective_value() << std::endl;
      else
        std::cerr << "\nMathOpt solve failed: " << result.status() << std::endl;
    }
  }

  // --- MPSolver x10 ---
  for (int i = 0; i < kRuns; ++i) {
    MPSolver solver("mpsolver_test", MPSolver::GLOP_LINEAR_PROGRAMMING);
    std::string error;
    solver.LoadModelFromProto(mp_proto, &error);
    auto start = std::chrono::high_resolution_clock::now();
    const auto status = solver.Solve();
    auto end = std::chrono::high_resolution_clock::now();
    mpsolver_times.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    if (i == 0) {
      if (status == MPSolver::OPTIMAL)
        std::cout << "MPSolver optimal value: " << solver.Objective().Value() << std::endl;
      else
        std::cerr << "MPSolver solve failed: " << status << std::endl;
    }
  }

  // --- Results ---
  auto print_stats = [](const std::string& name, const std::vector<long long>& times) {
    long long sum = std::accumulate(times.begin(), times.end(), 0LL);
    long long mn = *std::min_element(times.begin(), times.end());
    long long mx = *std::max_element(times.begin(), times.end());
    double avg = static_cast<double>(sum) / times.size();

    std::cout << "\n=== " << name << " (" << times.size() << " runs) ===" << std::endl;
    for (int i = 0; i < (int)times.size(); ++i)
      std::cout << "  run " << i + 1 << ": " << times[i] << " µs" << std::endl;
    std::cout << "  min: " << mn << " µs  max: " << mx
              << " µs  avg: " << avg << " µs" << std::endl;
  };

  print_stats("MathOpt (GLOP)", mathopt_times);
  print_stats("MPSolver (GLOP)", mpsolver_times);

  return 0;
}

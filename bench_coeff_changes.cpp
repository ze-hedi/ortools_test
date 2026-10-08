#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ortools/math_opt/cpp/math_opt.h"
#include "ortools/math_opt/io/mps_converter.h"

namespace math_opt = operations_research::math_opt;

const std::string kMpsFile =
    "/home/bouchehdahed/studies/clean_big_study/15-15_1000/sub/sub_15.mps";

const std::vector<int> kBatchSizes = {10000, 30000, 50000, 100000};
constexpr int kRuns = 10;

math_opt::SolverType SolverTypeFromString(const std::string& name) {
  std::string lower = name;
  for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lower == "glop") return math_opt::SolverType::kGlop;
  if (lower == "highs") return math_opt::SolverType::kHighs;
  if (lower == "pdlp" || lower == "pdl") return math_opt::SolverType::kPdlp;
  if (lower == "gscip" || lower == "scip") return math_opt::SolverType::kGscip;
  if (lower == "xpress") return math_opt::SolverType::kXpress;
  std::cerr << "Unknown solver: " << name
            << " (valid: glop, highs, pdlp, gscip, xpress)" << std::endl;
  std::exit(1);
}

struct CoeffEntry {
  math_opt::LinearConstraint constraint;
  math_opt::Variable variable;
  double original_value;
  CoeffEntry(math_opt::LinearConstraint c, math_opt::Variable v, double val)
      : constraint(c), variable(v), original_value(val) {}
};

int main(int argc, char* argv[]) {
  const std::string solver_name = argc > 1 ? argv[1] : "xpress";
  const math_opt::SolverType solver_type = SolverTypeFromString(solver_name);
  std::cout << "Solver: " << solver_name << std::endl;

  std::ofstream csv(solver_name + ".csv");
  if (!csv.is_open()) {
    std::cerr << "Failed to open " << solver_name << ".csv" << std::endl;
    return 1;
  }
  csv << "batch_size,set_avg_us,solve_avg_us\n";

  // Quick initial load just to print dimensions
  {
    auto proto_or = math_opt::ReadMpsFile(kMpsFile);
    if (!proto_or.ok()) {
      std::cerr << "Failed to read MPS: " << proto_or.status() << std::endl;
      return 1;
    }
    auto model_or = math_opt::Model::FromModelProto(*proto_or);
    if (!model_or.ok()) {
      std::cerr << "Failed to build model: " << model_or.status() << std::endl;
      return 1;
    }
    auto& model = *model_or;
    std::cout << "Model: " << kMpsFile << std::endl;
    std::cout << "Variables: " << model->num_variables()
              << "  Constraints: " << model->num_linear_constraints() << std::endl;
    size_t nnz = 0;
    for (const auto& ct : model->SortedLinearConstraints())
      nnz += model->RowNonzeros(ct).size();
    std::cout << "Total nonzero coefficients: " << nnz << std::endl;
  }

  // Baseline solve (also from scratch)
  {
    std::cout << "\n=== Baseline (no changes) ===" << std::endl;
    std::vector<long long> times;
    for (int run = 0; run < kRuns; ++run) {
      auto proto_or = math_opt::ReadMpsFile(kMpsFile);
      auto model_or = math_opt::Model::FromModelProto(*proto_or);
      auto& mdl = *model_or;
      auto start = std::chrono::high_resolution_clock::now();
      auto result = math_opt::Solve(*mdl, solver_type);
      auto end = std::chrono::high_resolution_clock::now();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times.push_back(us);
      std::cout << "  run " << run + 1 << ": solve=" << us << " µs";
      if (run == 0 && result.ok())
        std::cout << "  obj=" << result->objective_value();
      else if (run == 0)
        std::cout << "  FAILED";
      std::cout << std::endl;
    }
    double avg = static_cast<double>(std::accumulate(times.begin(), times.end(), 0LL)) / times.size();
    std::cout << "  avg: " << avg << " µs (" << avg / 1000.0 << " ms)" << std::endl;
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> perturbation(0.9, 1.1);

  for (int batch_size : kBatchSizes) {
    std::cout << "\n=== Batch size: " << batch_size << " coefficient changes ===" << std::endl;

    std::vector<long long> times, set_times;
    for (int run = 0; run < kRuns; ++run) {
      // Re-read MPS from disk
      auto proto_or = math_opt::ReadMpsFile(kMpsFile);
      if (!proto_or.ok()) {
        std::cerr << "  Failed to read MPS on run " << run << std::endl;
        continue;
      }
      auto model_or = math_opt::Model::FromModelProto(*proto_or);
      if (!model_or.ok()) {
        std::cerr << "  Failed to build model on run " << run << std::endl;
        continue;
      }
      auto& mdl = *model_or;

      // Collect all nonzero coefficients
      std::vector<CoeffEntry> coeffs;
      for (const auto& ct : mdl->SortedLinearConstraints()) {
        for (const auto& var : mdl->RowNonzeros(ct)) {
          coeffs.emplace_back(ct, var, mdl->coefficient(ct, var));
        }
      }

      if (batch_size > (int)coeffs.size()) {
        std::cout << "  Skipping (only " << coeffs.size() << " coefficients)" << std::endl;
        break;
      }

      // Sample random indices
      std::vector<int> indices(coeffs.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), rng);

      // Apply perturbations and time it
      auto set_start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < batch_size; ++i) {
        auto& e = coeffs[indices[i]];
        mdl->set_coefficient(e.constraint, e.variable, e.original_value * perturbation(rng));
      }
      auto set_end = std::chrono::high_resolution_clock::now();
      set_times.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(set_end - set_start).count());

      // Solve and time
      auto start = std::chrono::high_resolution_clock::now();
      auto result = math_opt::Solve(*mdl, solver_type);
      auto end = std::chrono::high_resolution_clock::now();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times.push_back(us);
      std::cout << "  run " << run + 1 << ": set=" << set_times.back()
                << " µs  solve=" << us << " µs" << std::endl;
    }

    if (times.empty()) continue;

    double avg = static_cast<double>(std::accumulate(times.begin(), times.end(), 0LL)) / times.size();
    double set_avg = static_cast<double>(std::accumulate(set_times.begin(), set_times.end(), 0LL)) / set_times.size();
    std::cout << "  set_coefficient avg: " << set_avg << " µs" << std::endl;
    std::cout << "  solve avg:           " << avg << " µs (" << avg / 1000.0 << " ms)" << std::endl;
    csv << batch_size << "," << set_avg << "," << avg << "\n";
  }

  return 0;
}

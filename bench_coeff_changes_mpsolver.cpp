#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ortools/linear_solver/linear_solver.h"
#include "ortools/lp_data/mps_reader.h"

const std::string kMpsFile =
    "/home/bouchehdahed/studies/clean_big_study/15-15_1000/sub/sub_15.mps";

const std::vector<int> kBatchSizes = {10000, 30000, 50000, 100000};
constexpr int kRuns = 10;

std::string SolverIdFromName(const std::string& name) {
  std::string lower = name;
  for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lower == "glop") return "glop";
  if (lower == "highs") return "highs";
  if (lower == "pdlp" || lower == "pdl") return "pdlp";
  if (lower == "gscip" || lower == "scip") return "scip";
  if (lower == "xpress") return "xpress";
  std::cerr << "Unknown solver: " << name
            << " (valid: glop, highs, pdlp, gscip, xpress)" << std::endl;
  std::exit(1);
}

struct CoeffEntry {
  operations_research::MPConstraint* constraint;
  const operations_research::MPVariable* variable;
  double original_value;
  CoeffEntry(operations_research::MPConstraint* c,
             const operations_research::MPVariable* v, double val)
      : constraint(c), variable(v), original_value(val) {}
};

int main(int argc, char* argv[]) {
  const std::string solver_name = argc > 1 ? argv[1] : "xpress";
  const std::string solver_id = SolverIdFromName(solver_name);
  std::cout << "Solver: " << solver_name << " (" << solver_id << ")" << std::endl;

  std::ofstream csv(solver_name + "_mpsolver.csv");
  if (!csv.is_open()) {
    std::cerr << "Failed to open " << solver_name << "_mpsolver.csv" << std::endl;
    return 1;
  }
  csv << "batch_size,set_avg_us,solve_avg_us\n";

  // Read the MPS model once into a proto; each run loads a fresh solver from it.
  auto proto_or = operations_research::glop::MpsFileToMPModelProto(kMpsFile);
  if (!proto_or.ok()) {
    std::cerr << "Failed to read MPS: " << proto_or.status() << std::endl;
    return 1;
  }
  const auto& model_proto = *proto_or;

  size_t nnz = 0;
  for (const auto& ct : model_proto.constraint()) nnz += ct.coefficient_size();
  std::cout << "Model: " << kMpsFile << std::endl;
  std::cout << "Variables: " << model_proto.variable_size()
            << "  Constraints: " << model_proto.constraint_size() << std::endl;
  std::cout << "Total nonzero coefficients: " << nnz << std::endl;

  // Baseline solve (also from scratch)
  {
    std::cout << "\n=== Baseline (no changes) ===" << std::endl;
    std::vector<long long> times;
    for (int run = 0; run < kRuns; ++run) {
      std::unique_ptr<operations_research::MPSolver> solver(
          operations_research::MPSolver::CreateSolver(solver_id));
      if (!solver) {
        std::cerr << "Failed to create solver: " << solver_id << std::endl;
        return 1;
      }
      std::string error;
      if (solver->LoadModelFromProto(model_proto, &error) !=
          operations_research::MPSOLVER_MODEL_IS_VALID) {
        std::cerr << "Failed to load model: " << error << std::endl;
        return 1;
      }
      auto start = std::chrono::high_resolution_clock::now();
      auto status = solver->Solve();
      auto end = std::chrono::high_resolution_clock::now();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times.push_back(us);
      std::cout << "  run " << run + 1 << ": solve=" << us << " µs";
      if (run == 0 && status == operations_research::MPSolver::OPTIMAL)
        std::cout << "  obj=" << solver->Objective().Value();
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
      std::unique_ptr<operations_research::MPSolver> solver(
          operations_research::MPSolver::CreateSolver(solver_id));
      if (!solver) {
        std::cerr << "  Failed to create solver on run " << run << std::endl;
        continue;
      }
      std::string error;
      if (solver->LoadModelFromProto(model_proto, &error) !=
          operations_research::MPSOLVER_MODEL_IS_VALID) {
        std::cerr << "  Failed to load model on run " << run << ": " << error << std::endl;
        continue;
      }

      // Collect all nonzero coefficients
      std::vector<CoeffEntry> coeffs;
      for (auto* ct : solver->constraints()) {
        for (const auto& [var, value] : ct->terms()) {
          coeffs.emplace_back(ct, var, value);
        }
      }

      if (batch_size > static_cast<int>(coeffs.size())) {
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
        e.constraint->SetCoefficient(e.variable, e.original_value * perturbation(rng));
      }
      auto set_end = std::chrono::high_resolution_clock::now();
      set_times.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(set_end - set_start).count());

      // Solve and time
      auto start = std::chrono::high_resolution_clock::now();
      auto status = solver->Solve();
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

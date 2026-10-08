#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/time/time.h"
#include "ortools/math_opt/cpp/math_opt.h"
#include "ortools/math_opt/io/mps_converter.h"

namespace math_opt = operations_research::math_opt;

struct CoeffEntry {
  math_opt::LinearConstraint constraint;
  math_opt::Variable variable;
  double original_value;
  CoeffEntry(math_opt::LinearConstraint c, math_opt::Variable v, double val)
      : constraint(c), variable(v), original_value(val) {}
};

struct BenchResult {
  double initial_build_ms = 0.0;
  double first_solve_ms = 0.0;
  double second_solve_ms = 0.0;
  double update_duration_ms = 0.0;
  int first_simplex_iters = 0;
  int first_barrier_iters = 0;
  int second_simplex_iters = 0;
  int second_barrier_iters = 0;
};

struct SolverBench {
  std::string solver_name;
  math_opt::SolverType solver_type;
  std::vector<BenchResult> runs;
};

BenchResult RunSingleRun(const std::string& mps_file,
                         math_opt::SolverType solver_type,
                         const std::string& perturb_mode) {
  BenchResult result;

  auto proto_or = math_opt::ReadMpsFile(mps_file);
  if (!proto_or.ok()) return result;
  auto model_or = math_opt::Model::FromModelProto(*proto_or);
  if (!model_or.ok()) return result;
  auto& model = *model_or;

  // Collect all nonzero coefficients
  std::vector<CoeffEntry> coeffs;
  for (const auto& ct : model->SortedLinearConstraints()) {
    for (const auto& var : model->RowNonzeros(ct)) {
      coeffs.emplace_back(ct, var, model->coefficient(ct, var));
    }
  }

  // Create incremental solver
  auto solver_or = math_opt::NewIncrementalSolver(model.get(), solver_type);
  if (!solver_or.ok()) return result;
  auto& inc_solver = *solver_or;

  result.initial_build_ms =
      absl::ToDoubleMicroseconds(inc_solver->initial_build_duration) / 1000.0;

  // First solve
  absl::StatusOr<math_opt::SolveResult> first_solve_result;
  {
    auto start = std::chrono::high_resolution_clock::now();
    first_solve_result = inc_solver->Solve();
    auto end = std::chrono::high_resolution_clock::now();
    result.first_solve_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
    if (first_solve_result.ok()) {
      result.first_simplex_iters = first_solve_result->solve_stats.simplex_iterations;
      result.first_barrier_iters = first_solve_result->solve_stats.barrier_iterations;
    }
  }

  // Apply 1000 random perturbations
  constexpr int kNumChanges = 1000;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> perturbation(0.9, 1.1);

  if (perturb_mode == "obj") {
    auto variables = model->SortedVariables();
    std::uniform_int_distribution<int> var_dist(
        0, static_cast<int>(variables.size()) - 1);
    for (int i = 0; i < kNumChanges; ++i) {
      auto var = variables[var_dist(rng)];
      double c = model->objective_coefficient(var);
      if (c != 0.0) {
        model->set_objective_coefficient(var, c * perturbation(rng));
      }
    }
  } else if (perturb_mode == "rhs") {
    auto constraints = model->SortedLinearConstraints();
    std::uniform_int_distribution<int> ct_dist(
        0, static_cast<int>(constraints.size()) - 1);
    for (int i = 0; i < kNumChanges; ++i) {
      auto ct = constraints[ct_dist(rng)];
      double ub = model->upper_bound(ct);
      if (std::isfinite(ub)) {
        model->set_upper_bound(ct, ub * perturbation(rng));
      }
      double lb = model->lower_bound(ct);
      if (std::isfinite(lb)) {
        model->set_lower_bound(ct, lb * perturbation(rng));
      }
    }
  } else if (perturb_mode != "basis") {
    // Default: coeff perturbation
    std::uniform_int_distribution<int> index_dist(
        0, static_cast<int>(coeffs.size()) - 1);
    for (int i = 0; i < kNumChanges; ++i) {
      auto& e = coeffs[index_dist(rng)];
      model->set_coefficient(e.constraint, e.variable,
                             e.original_value * perturbation(rng));
    }
  }
  // basis mode: no model perturbation, we just get/set the basis

  // Extract basis from first solve if available
  std::optional<math_opt::Basis> saved_basis;
  if (perturb_mode == "basis" && first_solve_result.ok() &&
      !first_solve_result->solutions.empty() &&
      first_solve_result->solutions[0].basis.has_value()) {
    saved_basis = first_solve_result->solutions[0].basis;
  }

  // Second solve (incremental)
  {
    math_opt::SolveArguments solve_args;
    if (saved_basis.has_value()) {
      solve_args.model_parameters.initial_basis = *saved_basis;
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto solve_result = inc_solver->Solve(solve_args);
    auto end = std::chrono::high_resolution_clock::now();
    result.second_solve_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
    result.update_duration_ms =
        absl::ToDoubleMicroseconds(inc_solver->last_update_duration) / 1000.0;
    if (solve_result.ok()) {
      result.second_simplex_iters = solve_result->solve_stats.simplex_iterations;
      result.second_barrier_iters = solve_result->solve_stats.barrier_iterations;
    }
  }

  return result;
}

double Average(const std::vector<BenchResult>& runs,
               double BenchResult::*field) {
  double sum = 0.0;
  for (const auto& r : runs) sum += r.*field;
  return sum / runs.size();
}

double AverageInt(const std::vector<BenchResult>& runs,
                  int BenchResult::*field) {
  double sum = 0.0;
  for (const auto& r : runs) sum += r.*field;
  return sum / runs.size();
}

void PrintSolverResults(const SolverBench& bench) {
  const int w = 16;

  std::cout << "\n--- " << bench.solver_name << " (" << bench.runs.size()
            << " runs) ---" << std::endl;

  std::cout << std::right << std::setw(w) << "Init Build" << std::setw(w)
            << "1st Solve" << std::setw(w) << "2nd Solve" << std::setw(w)
            << "Update" << std::setw(w) << "Simplex1" << std::setw(w)
            << "Barrier1" << std::setw(w) << "Simplex2" << std::setw(w)
            << "Barrier2" << std::endl;
  std::cout << std::string(8 * w, '-') << std::endl;

  for (size_t i = 0; i < bench.runs.size(); ++i) {
    const auto& r = bench.runs[i];
    std::cout << std::fixed << std::setprecision(2) << std::setw(w)
              << r.initial_build_ms << std::setw(w) << r.first_solve_ms
              << std::setw(w) << r.second_solve_ms << std::setw(w)
              << r.update_duration_ms << std::setw(w)
              << r.first_simplex_iters << std::setw(w)
              << r.first_barrier_iters << std::setw(w)
              << r.second_simplex_iters << std::setw(w)
              << r.second_barrier_iters << std::endl;
  }

  std::cout << std::string(8 * w, '-') << std::endl;
  std::cout << std::fixed << std::setprecision(2) << std::setw(w)
            << Average(bench.runs, &BenchResult::initial_build_ms)
            << std::setw(w)
            << Average(bench.runs, &BenchResult::first_solve_ms)
            << std::setw(w)
            << Average(bench.runs, &BenchResult::second_solve_ms)
            << std::setw(w)
            << Average(bench.runs, &BenchResult::update_duration_ms)
            << std::setw(w)
            << AverageInt(bench.runs, &BenchResult::first_simplex_iters)
            << std::setw(w)
            << AverageInt(bench.runs, &BenchResult::first_barrier_iters)
            << std::setw(w)
            << AverageInt(bench.runs, &BenchResult::second_simplex_iters)
            << std::setw(w)
            << AverageInt(bench.runs, &BenchResult::second_barrier_iters)
            << "  (avg)" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <mps_file> [num_runs=10] [coeff|rhs|obj|basis]" << std::endl;
    return 1;
  }

  absl::SetStderrThreshold(absl::LogSeverity::kError);
  absl::InitializeLog();

  const std::string mps_file = argv[1];
  const int num_runs = argc > 2 ? std::atoi(argv[2]) : 10;
  const std::string perturb_mode =
      argc > 3 ? std::string(argv[3]) : "coeff";

  std::cout << "Model: " << mps_file << std::endl;
  std::cout << "Runs:  " << num_runs << std::endl;
  std::cout << "Mode:  " << perturb_mode << std::endl;

  // Print dimensions once
  {
    auto proto_or = math_opt::ReadMpsFile(mps_file);
    auto model_or = math_opt::Model::FromModelProto(*proto_or);
    auto& model = *model_or;
    std::cout << "Variables: " << model->num_variables()
              << "  Constraints: " << model->num_linear_constraints()
              << std::endl;
    size_t nnz = 0;
    for (const auto& ct : model->SortedLinearConstraints())
      nnz += model->RowNonzeros(ct).size();
    std::cout << "Nonzeros: " << nnz << std::endl;
  }

  std::vector<SolverBench> solvers = {
      {"GLOP", math_opt::SolverType::kGlop, {}},
      {"HIGHS", math_opt::SolverType::kHighs, {}},
      {"XPRESS", math_opt::SolverType::kXpress, {}},
  };

  for (auto& sb : solvers) {
    for (int i = 0; i < num_runs; ++i) {
      sb.runs.push_back(RunSingleRun(mps_file, sb.solver_type, perturb_mode));
    }
    PrintSolverResults(sb);
  }

  // Write JSON
  const std::string json_file = "incremental_bench_" + perturb_mode + ".json";
  std::ofstream json(json_file);
  json << "{\n";
  for (size_t s = 0; s < solvers.size(); ++s) {
    const auto& sb = solvers[s];
    json << "  \"" << sb.solver_name << "\": {\n";
    json << std::fixed << std::setprecision(2);
    json << "    \"initial_build_ms\": "
         << Average(sb.runs, &BenchResult::initial_build_ms) << ",\n";
    json << "    \"first_solve_ms\": "
         << Average(sb.runs, &BenchResult::first_solve_ms) << ",\n";
    json << "    \"second_solve_ms\": "
         << Average(sb.runs, &BenchResult::second_solve_ms) << ",\n";
    json << "    \"update_duration_ms\": "
         << Average(sb.runs, &BenchResult::update_duration_ms) << ",\n";
    json << std::setprecision(0);
    json << "    \"first_simplex_iters\": "
         << AverageInt(sb.runs, &BenchResult::first_simplex_iters) << ",\n";
    json << "    \"first_barrier_iters\": "
         << AverageInt(sb.runs, &BenchResult::first_barrier_iters) << ",\n";
    json << "    \"second_simplex_iters\": "
         << AverageInt(sb.runs, &BenchResult::second_simplex_iters) << ",\n";
    json << "    \"second_barrier_iters\": "
         << AverageInt(sb.runs, &BenchResult::second_barrier_iters) << "\n";
    json << std::setprecision(2);
    json << "  }" << (s + 1 < solvers.size() ? "," : "") << "\n";
  }
  json << "}\n";
  std::cout << "\nResults written to " << json_file << std::endl;

  return 0;
}

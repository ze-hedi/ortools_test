#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include <vector>

int main() {
  std::cout << "=== Eigen bfloat16 Demo ===\n\n";

  // Basic conversion: float32 -> bfloat16 -> float32
  float original = 3.14159f;
  Eigen::bfloat16 bf = Eigen::bfloat16(original);
  float back = static_cast<float>(bf);
  std::cout << "float32:  " << original << "\n";
  std::cout << "bfloat16: " << back << "  (precision loss expected)\n";
  std::cout << "error:    " << std::abs(original - back) << "\n\n";

  // Arithmetic
  Eigen::bfloat16 a(2.5f), b(1.25f);
  Eigen::bfloat16 sum = Eigen::bfloat16(static_cast<float>(a) + static_cast<float>(b));
  Eigen::bfloat16 prod = Eigen::bfloat16(static_cast<float>(a) * static_cast<float>(b));
  std::cout << "a = " << static_cast<float>(a) << ", b = " << static_cast<float>(b) << "\n";
  std::cout << "a + b = " << static_cast<float>(sum) << "\n";
  std::cout << "a * b = " << static_cast<float>(prod) << "\n\n";

  // Size comparison
  std::cout << "sizeof(float):           " << sizeof(float) << " bytes\n";
  std::cout << "sizeof(Eigen::bfloat16): " << sizeof(Eigen::bfloat16) << " bytes\n";
  std::cout << "Memory saving:           " << (1.0 - (double)sizeof(Eigen::bfloat16) / sizeof(float)) * 100 << "%\n\n";

  // Simulate a small weight vector (like a transformer layer)
  constexpr int N = 8;
  std::vector<float> weights_f32 = {0.0012f, -0.523f, 1.337f, -0.0001f, 2.71f, -1.41f, 0.577f, -0.693f};
  std::vector<Eigen::bfloat16> weights_bf16(N);

  std::cout << "=== Simulated Weight Quantization (f32 -> bf16) ===\n";
  double total_err = 0;
  for (int i = 0; i < N; ++i) {
    weights_bf16[i] = Eigen::bfloat16(weights_f32[i]);
    float roundtrip = static_cast<float>(weights_bf16[i]);
    double err = std::abs(weights_f32[i] - roundtrip);
    total_err += err;
    std::cout << "  w[" << i << "] f32=" << weights_f32[i]
              << "  bf16=" << roundtrip
              << "  err=" << err << "\n";
  }
  std::cout << "  avg abs error: " << total_err / N << "\n";
  std::cout << "  memory: " << N * sizeof(float) << "B (f32) vs "
            << N * sizeof(Eigen::bfloat16) << "B (bf16)\n";

  return 0;
}

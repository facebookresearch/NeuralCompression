#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision) {
  std::vector<uint32_t> cdf(pmf.size() + 1);

  cdf[0] = 0;

  std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                 [=](float p) { return std::round(p * (1 << precision)); });

  const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);

  if (total == 0)
    throw std::domain_error("");

  std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                 [precision, total](uint32_t p) {
                   return ((static_cast<uint64_t>(1 << precision) * p) / total);
                 });

  std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());

  cdf.back() = 1 << precision;

  for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
    if (cdf[i] == cdf[i + 1]) {
      uint32_t peak = ~0u;

      int best = -1;

      for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
        uint32_t frequency = cdf[j + 1] - cdf[j];

        if (frequency > 1 && frequency < peak) {
          peak = frequency;
          best = j;
        }
      }

      assert(best != -1);

      if (best < i) {
        for (int j = best + 1; j <= i; ++j) {
          cdf[j]--;
        }
      } else {
        assert(best > i);

        for (int j = i + 1; j <= best; ++j) {
          cdf[j]++;
        }
      }
    }
  }

  assert(cdf[0] == 0);

  assert(cdf.back() == (1 << precision));

  for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
    assert(cdf[i + 1] > cdf[i]);
  }

  return cdf;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf, "");
}

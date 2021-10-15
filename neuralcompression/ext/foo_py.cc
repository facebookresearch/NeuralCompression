#include <torch/extension.h>

#include <iostream>

torch::Tensor foo() {
    return torch::rand({28, 28});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("foo", &foo, "");
}

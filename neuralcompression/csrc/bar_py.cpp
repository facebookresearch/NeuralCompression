#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bar() {
    return;
}

PYBIND11_MODULE(_bar, m) {
    m.def("bar", &bar, "");
}

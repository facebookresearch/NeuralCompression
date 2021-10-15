#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void foo() {
    return;
}

PYBIND11_MODULE(_foo, m) {
    m.def("foo", &foo, "");
}

#include "hnsw.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(hnsw_cpp, m) {
  m.doc() = "Python bindings for the HNSW C++ implementation";

  py::class_<Neighbor>(m, "Neighbor")
      .def_readonly("id", &Neighbor::id)
      .def_readonly("dist", &Neighbor::dist);

  py::class_<HNSWIndex>(m, "HNSWIndex")
      .def(py::init<int, int, int>(), py::arg("dim"), py::arg("M"),
           py::arg("ef_construction"))
      .def("add", &HNSWIndex::add, py::arg("vector"))
      .def(
          "search",
          [](const HNSWIndex &index, const Vec &query, int k, int ef_search) {
            auto results = index.search(query, k, ef_search);

            std::vector<std::pair<VecId, float>> output;
            output.reserve(results.size());

            for (const auto &result : results) {
              output.push_back({result.id, result.dist});
            }

            return output;
          },
          py::arg("query"), py::arg("k"), py::arg("ef_search"));
  ;
}

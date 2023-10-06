#include "IO.h"
#include "doubling_dpc.h"
#include "dpc_components.h"
#include "dpc_framework.h"
#include "utils.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <sstream>

namespace nb = nanobind;

using namespace nb::literals;

// TODO(Josh): Make data requirment more flexible? e.g. allow doubles or ints
DPC::ClusteringResult
dpc_numpy(nb::ndarray<float, nb::shape<nb::any, nb::any>, nb::device::cpu,
                      nb::c_contig>
              data,
          const unsigned K, const unsigned L, const unsigned Lnn,
          const std::shared_ptr<DPC::CenterFinder<double>> &center_finder,
          const std::string &output_path,
          const std::string &decision_graph_path, const unsigned Lbuild,
          const unsigned max_degree, const float alpha,
          const unsigned num_clusters, const std::string &method_str,
          const std::string &graph_type_str, bool use_new_framework) {

  float *data_ptr = data.data();
  size_t num_data = data.shape(0);
  size_t data_dim = data.shape(1);
  if (data_dim % 8 != 0) {
    throw std::invalid_argument(
        "For now, we only support data_dim that are a multiple of 8 for numpy "
        "data to avoid copies, so please pad your data if this is no the "
        "case.");
  }
  RawDataset raw_data(data_ptr, num_data, data_dim);

  DPC::Method method;
  std::istringstream(method_str) >> method;
  DPC::GraphType graph_type;
  std::istringstream(graph_type_str) >> graph_type;

  if (use_new_framework) {
    std::cout << "Using new DPC framework" << std::endl;
    return DPC::dpc_framework(K, L, Lnn, raw_data, center_finder, output_path,
                              decision_graph_path, Lbuild, max_degree, alpha,
                              num_clusters, method, graph_type);
  }

  // TODO: Deprecate this
  if (auto threshold_finder =
          dynamic_cast<DPC::ThresholdCenterFinder<double> *>(
              center_finder.get())) {
    return {DPC::dpc(K, L, Lnn, raw_data, density_cutoff, distance_cutoff,
                     center_density_cutoff, output_path, decision_graph_path,
                     Lbuild, max_degree, alpha, num_clusters, method,
                     graph_type),
            {}};
  }
  std::cout << "Using old DPC code" << std::endl;
}

DPC::ClusteringResult
dpc_filenames(const std::string &data_path, const unsigned K, const unsigned L,
              const unsigned Lnn, float density_cutoff,
              const std::shared_ptr<CenterFinder<double>> &center_finder,
              const std::string &decision_graph_path, const unsigned Lbuild,
              const unsigned max_degree, const float alpha,
              const unsigned num_clusters, const std::string &method_str,
              const std::string &graph_type_str, bool use_new_framework) {

  DPC::Method method;
  std::istringstream(method_str) >> method;
  DPC::GraphType graph_type;
  std::istringstream(graph_type_str) >> graph_type;

  RawDataset raw_data(data_path);

  DPC::ClusteringResult result;
  if (use_new_framework) {
    std::cout << "Using new DPC framework" << std::endl;
    result = DPC::dpc_framework(
        K, L, Lnn, raw_data, density_cutoff, distance_cutoff,
        center_density_cutoff, output_path, decision_graph_path, Lbuild,
        max_degree, alpha, num_clusters, method, graph_type);
  } else {
    // TODO: Deprecate this
    std::cout << "Using old DPC code" << std::endl;
    result = {DPC::dpc(K, L, Lnn, raw_data, density_cutoff, distance_cutoff,
                       center_density_cutoff, output_path, decision_graph_path,
                       Lbuild, max_degree, alpha, num_clusters, method,
                       graph_type),
              {}};
  }

  aligned_free(raw_data.data);

  return result;
}

NB_MODULE(dpc_ann_ext, m) {
  m.def("dpc_filenames", &dpc_filenames, "data_path"_a, "K"_a = 6, "L"_a = 12,
        "Lnn"_a = 4, "density_cutoff"_a = 0,
        "distance_cutoff"_a = std::numeric_limits<float>::max(),
        "center_density_cutoff"_a = 0, "output_path"_a = "",
        "decision_graph_path"_a = "", "Lbuild"_a = 12, "max_degree"_a = 16,
        "alpha"_a = 1.2, "num_clusters"_a = 4, "method"_a = "Doubling",
        "graph_type"_a = "Vamana", "use_new_framework"_a = true,
        "This function clusters the passed in numpy data and returns a "
        "ClusteringResult object with the clusters and metadata about the "
        "clustering process (including fine grained timing results).");

  // Don't want to allow conversion since then it will copy
  m.def("dpc_numpy", &dpc_numpy, "data"_a.noconvert(), "K"_a = 6, "L"_a = 12,
        "Lnn"_a = 4, "density_cutoff"_a = 0,
        "distance_cutoff"_a = std::numeric_limits<float>::max(),
        "center_density_cutoff"_a = 0, "output_path"_a = "",
        "decision_graph_path"_a = "", "Lbuild"_a = 12, "max_degree"_a = 16,
        "alpha"_a = 1.2, "num_clusters"_a = 4, "method"_a = "Doubling",
        "graph_type"_a = "Vamana", "use_new_framework"_a = true,
        "This function clusters the passed in files and returns a "
        "ClusteringResult object with the clusters and metadata about the "
        "clustering process (including fine grained timing results).");

  nb::class_<DPC::ClusteringResult>(m, "ClusteringResult")
      .def_rw("clusters", &DPC::ClusteringResult::clusters)
      .def_rw("metadata", &DPC::ClusteringResult::output_metadata);
}

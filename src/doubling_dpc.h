#pragma once

#include "utils.h"

namespace DPC {

void dpc(const unsigned K, const unsigned L, const unsigned Lnn,
         const std::string &data_path, float density_cutoff,
         float distance_cutoff, float center_density_cutoff,
         const std::string &output_path, const std::string &decision_graph_path,
         const unsigned Lbuild, const unsigned max_degree, const float alpha,
         const unsigned num_clusters, Method method, GraphType graph_type);

}
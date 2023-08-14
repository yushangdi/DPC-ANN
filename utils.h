#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>  // For boost::to_lower

#include "ParlayANN/algorithms/vamana/neighbors.h"
#include "ParlayANN/algorithms/utils/parse_files.h"
// #include "ParlayANN/algorithms/vamana/index.h"
// #include "ParlayANN/algorithms/utils/types.h"
// #include "ParlayANN/algorithms/utils/beamSearch.h"
// #include "ParlayANN/algorithms/utils/stats.h"


namespace DPC {

void report(double time, std::string str) {
    std::ios::fmtflags cout_settings = std::cout.flags();
    std::cout.precision(4);
    std::cout << std::fixed;
    // std::cout << name << ": ";
    if (str.length() > 0)
      std::cout << str << ": ";
    std::cout << time << std::endl;
    std::cout.flags(cout_settings);
}

enum class Method {
	Doubling, BlindProbe
};

// Overload the stream insertion operator for the Method enum class
std::ostream& operator<<(std::ostream& os, const Method& method) {
    switch (method) {
        case Method::Doubling:
            os << "Doubling";
            break;
        case Method::BlindProbe:
            os << "BlindProbe";
            break;
        default:
            os << "Unknown Method";
            break;
    }
    return os;
}

void validate(boost::any& v,
              const std::vector<std::string>& values,
              Method*, int) {
    namespace po = boost::program_options;

    po::validators::check_first_occurrence(v);

    const std::string& s = po::validators::get_single_string(values);
    std::string lower_s = s;
    boost::to_lower(lower_s);

    if (lower_s == "doubling") {
        v = Method::Doubling;
    } else if (lower_s == "blindprobe") {
        v = Method::BlindProbe;
    } else {
        throw po::validation_error(po::validation_error::invalid_option_value);
    }
}



template<class T>
void output(const std::vector<T>& densities, const std::vector<int>& cluster, const std::vector<std::pair<uint32_t, double>>& dep_ptrs, 
						const std::string& output_path, const std::string& decision_graph_path){
    if(output_path != ""){
    	std::ofstream fout(output_path);
    	for (size_t i = 0; i < cluster.size(); i++){
    		fout << cluster[i] << std::endl;
    	}
    	fout.close();
	}

	if(decision_graph_path != ""){    	
		std::cout << "writing decision graph\n";
    	std::ofstream fout(decision_graph_path);
    	for (size_t i = 0; i < densities.size(); i++){
    		fout << densities[i] << " " << dep_ptrs[i].second << '\n';
    	}
    }
}

template<typename T>
void writeVectorToFile(const std::vector<T>& vec, const std::string& filepath) {
    std::ofstream outFile(filepath);
    if (!outFile) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return;
    }
    for (const T& item : vec) {
        outFile << item << '\n';
    }
    outFile.close();
}



}
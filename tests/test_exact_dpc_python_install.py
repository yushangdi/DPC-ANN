import par_dpc
# from par_dpc import par_dpc_ext as par_dpc
import numpy as np
import sys
from pathlib import Path
import os

# Change to DPC-ANN folder and add to path
abspath = Path(__file__).resolve().parent.parent
os.chdir(abspath)
sys.path.append(str(abspath))

from post_processors import cluster_eval

def test_gaussian():
    query_file = "./data/gaussian_example/gaussian_4_1000.data"
    decision_graph_path = "./results/gaussian_4_1000.dg"
    output_path = "./results/gaussian_4_1000_numpy.cluster"
    gt_path = "./data/gaussian_example/gaussian_4_1000.gt"

    data = np.load("./data/gaussian_example/gaussian_4_1000.npy").astype("float64")
    times = par_dpc.dpc_sddp_numpy(
        data=data,
        decision_graph_path=decision_graph_path,
        output_path=output_path,
        depCut = 8.36
    )
    metrics1 = cluster_eval.eval_clusters_wrapper(gt_path, output_path, verbose=True)

    output_path = "./results/gaussian_4_1000_file.cluster"
    time_reports = par_dpc.dpc_sddp_filename(
        data_path=query_file,
        decision_graph_path=decision_graph_path,
        output_path=output_path,
        depCut = 8.36
    )
    print(time_reports)
    metrics2 = cluster_eval.eval_clusters_wrapper(gt_path, output_path, verbose=True)

    assert sorted(metrics1.items()) == sorted(metrics2.items())
    print(metrics1)
    print(metrics2)


if __name__ == "__main__":
    test_gaussian()

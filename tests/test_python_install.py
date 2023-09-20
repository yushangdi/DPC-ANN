import dpc_ann
import numpy as np
import sys
sys.path.append('/home/ubuntu/DPC-ANN')
from post_processors import cluster_eval

query_file = "./data/gaussian_example/gaussian_4_1000.data"
decision_graph_path = "./results/gaussian_4_1000.dg"
output_path = "./results/gaussian_4_1000.cluster"
gt_path = "./data/gaussian_example/gaussian_4_1000.gt"
graph_type = "Vamana"

data = np.load("./data/gaussian_example/gaussian_4_1000.npy").astype("float32")
print(data)
times = dpc_ann.dpc_numpy(
        distance_cutoff=8.36,
        data=data,
        decision_graph_path=decision_graph_path,
        output_path=output_path,
        graph_type=graph_type,
    )
print(times)
metrics1 = cluster_eval.eval_cluster_files(gt_path, output_path, verbose=True)

time_reports = dpc_ann.dpc_filenames(
            data_path=query_file,
            decision_graph_path=decision_graph_path,
            output_path=output_path,
            graph_type=graph_type,
            distance_cutoff=8.36
        )
print(time_reports)
metrics2 = cluster_eval.eval_cluster_files(gt_path, output_path, verbose=True)


print(metrics1)
print(metrics2)

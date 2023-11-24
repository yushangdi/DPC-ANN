import argparse
import pandas as pd
import matplotlib.pyplot as plt

gt_num_clusters = {
    "mnist": 10,
    "imagenet": 1000,
    "arxiv-clustering-s2s": 180,
    "reddit-clustering": 50,
    "birds": 525,
}


def plot_ari_by_cluster_offset_mult_figures(csv_path):
    plt.clf()
    df = pd.read_csv(csv_path)
    num_datasets = len(gt_num_clusters)
    num_cols = 3
    num_rows = (num_datasets + num_cols - 1) // num_cols
    plot_scaler = 6

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(plot_scaler * num_cols, plot_scaler * num_rows),
        tight_layout=True,
        sharex=True,
    )
    axes = axes.reshape(-1)

    dataset_groups = df.groupby("dataset")

    for (dataset_name, dataset_group), ax in zip(dataset_groups, axes):
        ax.set_title(dataset_name, fontsize=22)

        for method_name, method_group in dataset_group.groupby("method"):
            method_group["cluster_ratio"] = (
                method_group["num_clusters"] / gt_num_clusters[dataset_name]
            )
            ax.plot(
                method_group["cluster_ratio"],
                method_group["ARI"],
                label=f"{method_name}",
            )

    for i in range(num_datasets, num_rows * num_cols):
        axes[i].axis("off")

    plt.suptitle(
        'Effect of Clustering with the "Wrong" Number of Clusters', fontsize=20
    )
    plt.tight_layout()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.72, 0.3), fontsize=18)

    plt.savefig("results/paper/varying_num_clusters_all.pdf")


def plot_ari_by_cluster_offset_one_figure_ours(csv_path):
    plt.clf()

    df = pd.read_csv(csv_path)

    grouped_data = df.groupby(["dataset", "method"])

    for (dataset, method), group in grouped_data:
        if method != "Vamana":
            continue
        group["cluster_ratio"] = group["num_clusters"] / gt_num_clusters[dataset]

        plt.plot(group["cluster_ratio"], group["ARI"], label=f"{dataset}")

    plt.title('Effect of Clustering with the "Wrong" Number of Clusters')
    plt.axvline(1, c="black", linestyle=":")
    plt.text(1.1, 0, "Correct number of clusters", rotation=90)
    plt.xlabel("Cluster Ratio: # Clusters Used / # Clusters in Ground Truth")
    plt.ylabel("ARI")
    plt.legend(title="Dataset")
    plt.savefig("results/paper/varying_num_clusters_vamana.pdf")


def plot_homogeneity_vs_completeness_pareto(csv_path):
    plt.clf()

    df = pd.read_csv(csv_path)

    grouped_data = df.groupby(["dataset", "method"])

    for name, group in grouped_data:
        plt.scatter(
            group["homogeneity"].to_list() + [1, 0],
            group["completeness"].to_list() + [0, 1],
            label=f"{name}",
        )

    plt.title("Homogeneity vs Completeness Pareto")
    plt.xlabel("Homogeneity")
    plt.ylabel("Completeness")
    plt.legend(title="Dataset")
    plt.savefig("results/paper/varying_num_clusters_homogeneity_vs_completeness.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot ARI vs percent of the ground truth number of clusters"
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the csv containing the varying number of clusters experiment data",
    )

    args = parser.parse_args()

    plot_homogeneity_vs_completeness_pareto(args.csv_path)
    plot_ari_by_cluster_offset_one_figure_ours(args.csv_path)
    plot_ari_by_cluster_offset_mult_figures(args.csv_path)
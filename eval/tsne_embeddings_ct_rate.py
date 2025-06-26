import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Visualize projected embeddings with t-SNE"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to CSV file with VolumeName and abnormality flags"
    )
    parser.add_argument(
        "--embedding_dir", type=str, required=True,
        help="Directory where each sample subfolder holds embedding .npy files"
    )
    parser.add_argument(
        "--embedding_type", type=str, default="image_projection",
        help="Which .npy embedding to load (without extension)"
    )
    parser.add_argument(
        "--perplexity", type=float, default=50,
        help="Perplexity for t-SNE"
    )
    parser.add_argument(
        "--output_plot", type=str, default=None,
        help="Path to save the t-SNE plot (optional)"
    )
    return parser


def assign_group(n):
    if n == 0:
        return '0'
    elif 1 <= n <= 4:
        return '1-4'
    elif 5 <= n <= 8:
        return '5-8'
    else:
        return '9+'


def main(args):

    # Load labels
    df = pd.read_csv(args.csv_path)
    cols = df.columns.difference(["VolumeName"])
    df["num_abnormalities"] = (df[cols] == 1).sum(axis=1)
    df["group"] = df["num_abnormalities"].apply(assign_group)

     # Load embeddings
    embeddings = []
    missing = []
    for _, row in df.iterrows():
        fname = Path(row["VolumeName"]).stem
        emb_path = Path(args.embedding_dir) / fname / f"{args.embedding_type}.npy"
        if not emb_path.exists():
            missing.append(str(emb_path))
            continue
        emb = np.load(emb_path)
        embeddings.append(emb.flatten())

    if missing:
        print(f"Warning: {len(missing)} embeddings missing. Examples: {missing[:5]}")

    embeddings = np.array(embeddings)
    print(f"Loaded {len(embeddings)} embeddings for t-SNE.")

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=args.perplexity)
    tsne_result = tsne.fit_transform(embeddings)

    # Plot
    colors = {"0": "red", "1-4": "orange", "5-8": "blue", "9+": "green"}
    plt.figure(figsize=(10, 8))
    for grp in df["group"].unique():
        idxs = df.index[df["group"] == grp].tolist()
        plt.scatter(
            tsne_result[idxs, 0], tsne_result[idxs, 1],
            label=grp, color=colors.get(grp, "gray"), alpha=0.7, s=5
        )
    plt.title(f"t-SNE of {args.embedding_type} embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="# abnormalities")
    plt.grid(True)
    plt.tight_layout()

    if args.output_plot:
        plt.savefig(args.output_plot)
        print(f"Plot saved to {args.output_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

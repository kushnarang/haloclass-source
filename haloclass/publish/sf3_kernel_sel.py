from sklearn import svm
from pathlib import Path
from tap import Tap
from transformers import set_seed
import pandas as pd

from haloclass.publish.helpers import create_bootstrapped_metrics_table, load_pickled_dataset

class KernelSelArgs(Tap):
    data_path: str = "publication-datasets"

KERNELS = ["linear", "poly", "rbf", "sigmoid"]

def main():
    args = KernelSelArgs().parse_args()
    data_path = Path(args.data_path)
    training_x, training_y = load_pickled_dataset(data_path / "train.150.embeddings", data_path / "train.150.labels")
    eval_x, eval_y = load_pickled_dataset(data_path / "eval.150.embeddings", data_path / "eval.150.labels")

    set_seed(42)

    all_dfs = []
    for kernel in KERNELS:
        print(f"starting {kernel} evaluation")
        model = svm.SVC(probability=True, kernel=kernel)
        model.fit(training_x, training_y)
        df = create_bootstrapped_metrics_table(model, (eval_x, eval_y), extra_columns={"model": kernel})
        all_dfs.append(df)
    
    pd.concat(all_dfs).to_csv(f"{args.data_path}/sf/kernel_sel.csv", index=False)

if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
from sklearn.model_selection import ParameterGrid
from tap import Tap
from sklearn import svm
from transformers import set_seed
import pandas as pd
from tqdm import tqdm

from haloclass.publish.helpers import create_bootstrapped_metrics_table, load_pickled_dataset

class PlmSelectionArgs(Tap):
    data_path: str = "publication-datasets"

C_range = np.logspace(-1, 1, 3)
gamma_range = np.logspace(-6, 0, 3)

grid = ParameterGrid({
    "kernel": ["linear"],
    "C": C_range,
    "gamma": gamma_range,
    "random_state": [42],
})

def main():
    args = PlmSelectionArgs().parse_args()
    data_path = Path(args.data_path)
    training_x, training_y = load_pickled_dataset(data_path / "train.150.embeddings", data_path / "train.150.labels")
    eval_x, eval_y = load_pickled_dataset(data_path / "eval.150.embeddings", data_path / "eval.150.labels")

    set_seed(42)

    all_dfs = []
    for param_set in tqdm(grid):
        print(f"starting {param_set} evaluation")
        model = svm.SVC(probability=True, **param_set)
        model.fit(training_x, training_y)
        df = create_bootstrapped_metrics_table(model, (eval_x, eval_y), extra_columns={"model": param_set})
        all_dfs.append(df)
    
    pd.concat(all_dfs).to_csv(f"{args.data_path}/sf/hyper_sel.csv", index=False)


if __name__ == "__main__":
    main()
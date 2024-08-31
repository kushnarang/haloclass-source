from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import svm
from tap import Tap
from transformers import set_seed
import pandas as pd

from haloclass.publish.helpers import create_bootstrapped_metrics_table, load_pickled_dataset

all_checkpoints = ["8", "35", "150", "650"]

class PlmSelectionArgs(Tap):
    data_path: str = "publication-datasets"

def main():
    args = PlmSelectionArgs().parse_args()
    data_path = Path(args.data_path)

    set_seed(42)

    # training_datasets = [load_pickled_dataset(data_path / f"train.{cp}.embeddings", data_path / f"train.{cp}.labels") for cp in all_checkpoints]
    eval_datasets = [load_pickled_dataset(data_path / f"eval.{cp}.embeddings", data_path / f"eval.{cp}.labels") for cp in all_checkpoints]

    split_eval_dataset = [train_test_split(e, l, test_size=0.2, random_state=42) for e, l in eval_datasets]

    # verify all the datasets were shuffled the same way
    tmp_train_labels = [e[3] for e in split_eval_dataset]
    assert all(x == tmp_train_labels[0] for x in tmp_train_labels)

    all_dfs = []

    for cp, evaluation in zip(all_checkpoints, split_eval_dataset):
        X_train, X_eval, y_train, y_eval = evaluation
        assert len(X_eval) == len(y_eval)

        model = svm.SVC(probability=True)
        model.fit(X_train, y_train)
        df = create_bootstrapped_metrics_table(model, (X_eval, y_eval), extra_columns={"model": cp})
        print(df.head())
        all_dfs.append(df)

    pd.concat(all_dfs).to_csv(f"{args.data_path}/sf/plm_sel.csv", index=False)



if __name__ == "__main__":
    main()
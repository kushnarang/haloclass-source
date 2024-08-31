from pathlib import Path
from tap import Tap

from haloclass.publish.helpers import create_bootstrapped_metrics_table, load_pickled_dataset
import pandas as pd

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, preprocessing

class PlmSelectionArgs(Tap):
    data_path: str = "publication-datasets"


MODELS_TO_TEST = [("SVM", svm.SVC(probability=True)),
    ("XGBoost", xgb.XGBClassifier()),
    ("Logistic Regression", make_pipeline(preprocessing.StandardScaler(),LogisticRegression())),
    ("KNN", KNeighborsClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("MLP", MLPClassifier()),
    ("Gaussian Bayes", GaussianNB())]

def main():
    args = PlmSelectionArgs().parse_args()
    args.data_path = Path(args.data_path)
    training_x, training_y = load_pickled_dataset(args.data_path / "train.35.embeddings", args.data_path / "train.35.labels")
    eval_x, eval_y = load_pickled_dataset(args.data_path / "eval.35.embeddings", args.data_path / "eval.35.labels")

    all_dfs = []

    for name, model in MODELS_TO_TEST:
        print(f"Fitting {name}")
        model.fit(training_x, training_y)
        print(f"Done fitting {name}")
        df = create_bootstrapped_metrics_table(model, (eval_x, eval_y), extra_columns={"model": name})
        print(df.head())
        all_dfs.append(df)

    pd.concat(all_dfs).to_csv(args.data_path / "sf/arch_sel.csv", index=False)
    # indicies = [np.random.choice(np.arange(len(x_test)), size=len(x_test), replace=True) for _ in range(0, 100)]


if __name__ == "__main__":
    main()
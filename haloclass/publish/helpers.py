import pickle
from pathlib import Path

from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score, average_precision_score
import pandas as pd
import numpy as np
from tqdm import tqdm

def do_metrics(real, preds, preds_proba):
    roc = roc_auc_score(real, preds_proba[:, 1])
    mcc = matthews_corrcoef(real, preds)
    acc = accuracy_score(real, preds)
    auprc = average_precision_score(real, preds_proba[:, 1])

    tn, fp, fn, tp = confusion_matrix(real, preds).ravel()

    output = {
        "AUROC": roc,
        "MCC": mcc,
        "Accuracy": acc,
        "AUPRC": auprc,
        "True negatives": tn,
        "False positives": fp,
        "False negatives": fn,
        "True positives": tp
    }

    return output

def create_bootstrapped_metrics_table(model, dataset, n=100, extra_columns={}):
    x, y = dataset
    x = np.array(x)
    y = np.array(y)
    indicies = [np.random.choice(np.arange(len(x)), size=len(x), replace=True) for _ in range(0, n)]
    datasets = [(x[sel], y[sel]) for sel in indicies]

    results = []

    for i, d in enumerate(tqdm(datasets)):
        xs, ys = d
        metrics = do_metrics(ys, model.predict(xs), model.predict_proba(xs))
        metrics["iter"] = i
        results.append(metrics | extra_columns)

    df = pd.DataFrame(results)
    return df


def load_pickled_dataset(embeddings_src: Path, labels_str: Path):
    with open(embeddings_src, "rb") as f_embeddings:
        embeddings = pickle.load(f_embeddings)
    with open(labels_str, "rb") as f_labels:
        labels = pickle.load(f_labels)
    
    assert len(embeddings) == len(labels), "Embeddings and labels length don't match!"
    return embeddings, labels

def main():
    pass

if __name__ == "__main__":
    main()
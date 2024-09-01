import pickle
from pathlib import Path
from sklearn import svm
from tap import Tap


from haloclass.publish.helpers import load_pickled_dataset

class CreateModelArgs(Tap):
    data_path: str = "publication-datasets"
    embeddings_source: str = "train.150.embeddings"
    labels_source: str = "train.150.labels"
    save_name: str = 'publication-datasets/final-haloclass-model.pkl'


PARAMS = {
    "kernel": "linear",
    "C": 0.1,
    "gamma": 1e-06,
    "random_state": 42
}

def main():
    args = CreateModelArgs().parse_args()
    data_path = Path(args.data_path)
    training_x, training_y = load_pickled_dataset(data_path / args.embeddings_source, data_path / args.labels_source)

    model = svm.SVC(probability=True, **PARAMS)
    model.fit(training_x, training_y)

    with open(args.save_name, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
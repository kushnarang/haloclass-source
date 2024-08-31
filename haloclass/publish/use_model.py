
import pickle
import pandas as pd
from tap import Tap

from haloclass.publish.generate_embeddings import ImportedFasta, checkpoint150
from haloclass.publish.helpers import do_metrics, load_pickled_dataset
from pathlib import Path

class UseModelArgs(Tap):
    model_path: str = f"{Path(__file__).parent.parent.parent.absolute()}/publication-datasets/model.pkl"
    preset: str = None
    fasta: str = None
    save_preds: str = None
    ignore_labels: bool = True

PRESETS = {
    "zhang": ["publication-datasets/external/zhang139.fasta", lambda x: 1 if "Salinibacter ruber" in str(x.description) else 0],
    "siglioccolo": ["publication-datasets/external/siglioccolo.fasta", lambda x: 1 if "salt-in" in str(x.description) else 0],
    "newtestset": ["publication-datasets/test.150.embeddings", "publication-datasets/test.150.labels"],
    "tadeo": ["publication-datasets/external/tadeo.fasta", 1],
}

def evaluate_model(model, x, y):
    metrics = do_metrics(y, model.predict(x), model.predict_proba(x))
    return metrics

def main():
    args = UseModelArgs().parse_args()
    preset = args.preset
    fasta = args.fasta

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    combined = None

    if preset:
        if ".embeddings" in PRESETS[preset][0]:
            embeddings, labels = load_pickled_dataset(PRESETS[preset][0], PRESETS[preset][1])
        else:
            fasta_file, derive_label = PRESETS[preset]
            combined = ImportedFasta.from_fasta(fasta_file, derive_label)
            embeddings = checkpoint150(combined.sequences, combined.labels)
            labels = combined.labels
    elif fasta:
        combined = ImportedFasta.from_fasta(args.fasta, 0 if args.ignore_labels else "default")
        embeddings = checkpoint150(combined.sequences, combined.labels)
        labels = combined.labels
    else:
        raise "No valid input provided"

    if args.save_preds:
        sequences = combined.sequences if combined else None
        df = pd.DataFrame({ "sequences": sequences, "predictions": model.predict(embeddings), "confidences": model.predict_proba(embeddings)[:,1] })
        df.to_csv(args.save_preds, index=False)
        return

    print(evaluate_model(model, embeddings, labels))

if __name__ == "__main__":
    main()
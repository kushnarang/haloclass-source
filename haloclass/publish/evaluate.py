
import pickle
import pandas as pd
from tap import Tap

from haloclass.publish.generate_embeddings import ImportedFasta, checkpoint150
from pathlib import Path

from haloclass.publish.use_model import evaluate_model

class UseModelSimpleArgs(Tap):
    model_path: str = f"{Path(__file__).parent.parent.parent.absolute()}/publication-datasets/model.pkl"
    fasta: str
    save: str = "predictions.csv"
    print_perf: bool = False
    disable_accelerators: bool = False
    batch_size: int = 32

def main():
    args = UseModelSimpleArgs().parse_args()
    fasta = args.fasta
    save = args.save

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    
    sequence_derivation = "default" if args.print_perf else 0

    combined = ImportedFasta.from_fasta(fasta, sequence_derivation)
    embeddings = checkpoint150(combined.sequences, combined.labels, disable_accelerators=args.disable_accelerators, batch_size=args.batch_size)

    if args.print_perf:
        print("=" * 50)
        print("MODEL PERFORMANCE:")
        print(evaluate_model(model, embeddings, combined.labels))
        print("=" * 50)
    
    predictions = model.predict(embeddings)

    decisions = [("non-tolerant" if p == 0 else "salt-tolerant") for p in predictions]
    
    df = pd.DataFrame({ "sequences": combined.sequences, "predicted labels": decisions, "predictions": predictions, "confidences": model.predict_proba(embeddings)[:,1] })
    df.to_csv(save, index=False)
    print(f"Full model predictions are saved to {Path(save).absolute()}")

if __name__ == "__main__":
    main()
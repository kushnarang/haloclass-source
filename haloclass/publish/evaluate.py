
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
    perf_only: bool = False
    disable_accelerators: bool = False
    batch_size: int = 32

def main():
    args = UseModelSimpleArgs().parse_args()
    fasta = args.fasta
    save = args.save

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)
    
    combined = ImportedFasta.from_fasta(fasta, 0)
    embeddings = checkpoint150(combined.sequences, combined.labels, disable_accelerators=args.disable_accelerators, batch_size=args.batch_size)

    if args.perf_only:
        print(evaluate_model(model, embeddings, combined.labels))
        return
    
    df = pd.DataFrame({ "sequences": combined.sequences, "predictions": model.predict(embeddings), "confidences": model.predict_proba(embeddings)[:,1] })
    df.to_csv(save, index=False)

if __name__ == "__main__":
    main()
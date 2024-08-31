from tap import Tap
from pathlib import Path

from transformers import set_seed
from sklearn.model_selection import train_test_split

from haloclass.publish.generate_embeddings import ImportedFasta

class DatasetCreationArgs(Tap):
    input_file: str
    output_dir: str

def do_test_train_eval_split(sequences, labels):
    set_seed(42)
    train_ratio = 0.90
    validation_ratio = 0.05
    test_ratio = 0.05

    x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=1 - train_ratio, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)
    return ((x_train, y_train), (x_test, y_test), (x_val, y_val))

def save_to_file(path, sequences, labels):
    out = ""
    for s, l in zip(sequences, labels):
        out += f">{l}\n{s}\n"
    with open(path, "w") as f:
        f.write(out)

def main():
    args = DatasetCreationArgs().parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    fasta = ImportedFasta.from_fasta(input_file, "default")

    # Ensure correct labeling of sequences
    assert fasta.labels.count(0) == 17606
    assert fasta.labels.count(1) == 10424

    train_set, test_set, eval_set = do_test_train_eval_split(fasta.sequences, fasta.labels)

    output_dir.mkdir(exist_ok=True)

    save_to_file(output_dir / "train.fasta", *train_set)
    save_to_file(output_dir / "test.fasta", *test_set)
    save_to_file(output_dir / "eval.fasta", *eval_set)

if __name__ == "__main__":
    main()
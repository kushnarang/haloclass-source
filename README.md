# HaloClass: Salt-Tolerant Protein Classification with Protein Language Models

## Installation

HaloClass has been tested on Python 3.10.11.

`$ pip3 install typed-argument-parser torch transformers datasets biopython tqdm`
`$ pip3 install -e haloclass-source`
`$ cd haloclass-source`

## Usage

### To print performance metrics

True labels should be specified as the FASTA sequence names, see `publication-datasets/eval.fasta` for an example.

`$ python3 haloclass/publish/use_model.py --fasta INPUT_FASTA_FILE.fasta`

### To save predictions and confidences

A CSV file will be outputted with HaloClass predictions, salt tolerance confidence, and the corresponding sequence, in the same order as the sequences in the FASTA file. Labels in the FASTA sequence names will be ignored.

`$ python3 haloclass/publish/use_model.py --fasta INPUT_FASTA_FILE.fasta --save_preds OUTPUT_CSV_FILE.csv`
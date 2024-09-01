# HaloClass: Salt-Tolerant Protein Classification with Protein Language Models

Kush Narang*, Abhigyan Nath, William Hemstrom, Simon K. S. Chu

\* Corresponding author: knarang@ucdavis.edu

## Installation

HaloClass has been tested on Python 3.10.11.

`$ pip3 install typed-argument-parser torch transformers datasets biopython tqdm pandas numpy scikit-learn`

`$ pip3 install -e haloclass-source`

`$ cd haloclass-source`

## Usage


### To save predictions and confidences

A CSV file (`predictions.csv`) will be outputted with three columns: predicted labels, HaloClass predictions, HaloClass salt-tolerance confidences, and the corresponding sequences. Labels in the FASTA sequence names will be ignored.

`$ python3 haloclass/publish/evaluate.py --fasta INPUT_FASTA_FILE.fasta`

Interpreting confidences: A confidence of 0 indicates full confidence that the sequence is non-tolerant. A confidence of 1 indicates full confidence that the sequence is salt-tolerant. Intermediate values are how the model expresses non-certainty. Confidences are rounded to determine predictions and labels.


#### Changing the output file name (for evaluate.py)

To change the output CSV name (default is `predictions.csv`), use the flag:

`--save YOUR_CSV_NAME.csv`


#### Disabling hardware accelerators (for evaluate.py)

Out of the box, HaloClass supports MPS and CUDA accelerators. To disable them, use the flag:

`--disable_accelerators`

#### Changing batch size (for evaluate.py)

By default, HaloClass evaluates sequences in batches of 32. To alter this, use the flag:

`--batch_size YOUR_INTEGER_BATCH_SIZE`


---

### To print performance metrics

True labels should be specified as the FASTA sequence names `0` for **non-tolerant** and `1` for **salt-tolerant**. See `publication-datasets/eval.fasta` for an example. This flag will print the model's AUROC, MCC, AUPRC, Confusion Matrix stats, and Accuracy on this dataset. This relies on labels being _accurate and formatted correctly_.

`$ python3 haloclass/publish/evaluate.py --print_perf --fasta INPUT_FASTA_FILE.fasta`

---

### Retraining HaloClass

#### First, generate embeddings

`$ python3 haloclass/publish/generate_embeddings.py --input_file YOUR_FASTA_FILE.fasta --output_prefix YOUR_PREFIX`

This process will generate two output files:

1. `YOUR_PREFIX.150.embeddings`
2. `YOUR_PREFIX.150.labels`

Note, labels are taken from the FASTA sequence names (`0` for **non-tolerant** and `1` for **salt-tolerant**).

#### Second, train new model

`python3 haloclass/publish/create_model.py --data_path PATH --embeddings_source EMBEDS --labels_source LABELS --save_name YOUR_SAVE_NAME.pkl`

where `embeddings` and `labels` are saved as `PATH/EMBEDS` and `PATH/LABELS`, respectively.
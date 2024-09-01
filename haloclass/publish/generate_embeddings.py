from pathlib import Path
from tap import Tap
import torch
from transformers import EsmModel, AutoTokenizer, set_seed
from datasets import Dataset
import pickle

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from dataclasses import dataclass
import time
from tqdm import tqdm

class GenEmbedsArgs(Tap):
    input_file: str
    output_prefix: str

@dataclass
class ImportedFasta:
	sequences: list[SeqRecord]
	labels: list[int]
	def __post_init__(self):
		if len(self.labels) != len(self.sequences): raise ValueError("Length of labels and sequences don't match")
	@staticmethod
	def from_fasta(fasta_path: str, label) -> "ImportedFasta":
		with open(fasta_path) as f:
			sequences = [s for s in SeqIO.parse(f, "fasta")]

			if label == "default": label = [int(s.name.strip()) for s in sequences]
			if callable(label): label = [label(s) for s in sequences]
			
			sequences = [str(s.seq) for s in sequences]

			label = [label] * len(sequences) if type(label) == int else label

			# expand a pattern-style label argument
			if len(label) != len(sequences):
				label = label * (len(sequences) // len(label))
			
		return ImportedFasta(sequences, label)

def generate_embeddings(device, model, tokenizer, sequences, labels, batch_size=32):
	device = torch.device("cpu")
	if torch.backends.mps.is_available(): device = torch.device("mps")
	if torch.cuda.is_available(): device = torch.device("cuda")

	set_seed(42)
	start = time.time()
	def collate_fn(batch):
		tokenized_sequences = tokenizer(batch["sequences"], max_length=1022, truncation=True, padding='max_length')
		tokenized_sequences["labels"] = batch["labels"]
		return tokenized_sequences

	dataset = Dataset.from_dict({
		"labels": labels,
		"sequences": sequences,
		"lengths": [len(p) for p in sequences]
	}).map(collate_fn, remove_columns="sequences")

	loader = torch.utils.data.DataLoader(dataset.with_format("torch"), batch_size=batch_size)
	all_embeddings = []
	for _, data in enumerate(tqdm(loader)):
		with torch.no_grad():
			input_ids = data["input_ids"].to(device)
			attention_mask = data["attention_mask"].to(device)
			embedding_repr = model(input_ids, attention_mask=attention_mask)

			# embedding_repr.last_hidden_state shape: [batch_size, sequence, length, 640 (embedding size)]
			this_batch_size = embedding_repr.last_hidden_state.shape[0]

			for i in range(0, this_batch_size):
				this_length = data["lengths"][i]
				last_index = min(this_length+1, 1022)
				embedding = embedding_repr.last_hidden_state[i][1:last_index]

				embedding = embedding.mean(dim=0)
				assert embedding.shape[0] == 640
				
				all_embeddings.append(embedding)
	
	end = time.time()
	all_embeddings = [i.numpy(force=True) for i in all_embeddings]
	return all_embeddings, end - start

# 150M
def checkpoint150(sequences, labels, disable_accelerators=False, batch_size=32):
	device = torch.device("cpu")
	if not disable_accelerators:
		if torch.backends.mps.is_available(): device = torch.device("mps")
		if torch.cuda.is_available(): device = torch.device("cuda")
	
	print(f"Creating embeddings on {device}")

	set_seed(42)

	model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
	model = model.to(device)
	model = model.eval()

	tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
	embeddings, wall_time = generate_embeddings(device, model, tokenizer, sequences, labels, batch_size=batch_size)
	return embeddings

def main():
	args = GenEmbedsArgs().parse_args()

	input_file = Path(args.input_file)
	output_prefix = Path(args.output_prefix)

	combined = ImportedFasta.from_fasta(input_file, "default")

	embeddings = checkpoint150(combined.sequences, combined.labels)

	with open(output_prefix + ".150.embeddings", 'wb') as f:
		pickle.dump(embeddings, f)
	
	with open(output_prefix + ".150.labels", 'wb') as f:
		pickle.dump(combined.labels, f)

if __name__ == "__main__":
	main()
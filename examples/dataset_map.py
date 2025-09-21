from datasets import load_dataset
from ml_utils.data.sequence import TokenizeThenGroupBatchMap, TokenizingCollateFn
from ml_utils.data.base import batch_map_to_collator_format, collator_format_to_map_format

from transformers import AutoTokenizer

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")

print(dataset[0])

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


collate_fn = TokenizingCollateFn(tokenizer, max_seq_len=1024)

def map_fn(batch):
    batch = map_format_to_collator_format(batch)
    batch = collate_fn(batch)
    return {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
        "attention_mask": batch["attention_mask"],
        "text": batch["text"],
    }

tokenized_dataset = dataset.map(map_fn, batched=True)

print(dataset[1])



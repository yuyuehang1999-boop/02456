from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-103-v1")
ds.save_to_disk("wikitext-103")

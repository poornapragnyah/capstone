from transformers import BertTokenizer

# First time download → specify cache dir
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="/home/poorna/models")

# Save tokenizer locally
tokenizer.save_pretrained("/home/poorna/models/bert-base-uncased")


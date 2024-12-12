from transformers import AutoTokenizer
from collections import Counter
from nltk.util import ngrams
from crystalbleu import corpus_bleu
import pandas as pd
from tqdm import tqdm

import re




tokenizer = AutoTokenizer.from_pretrained(
    "../finetune/models/deepseek-coder-1.3b-instruct",
    trust_remote_code=True,
    pad_token="<｜end▁of▁sentence｜>",
    bos_token="<｜begin▁of▁sentence｜>",
    eos_token="<|EOT|>"
)


real_data = pd.read_csv("../finetune/data/model_predictions.csv")["Real Output"].tolist()
predicted_data = pd.read_csv("../finetune/data/model_predictions.csv")["Predicted Output"].tolist()


def tokenize_sentences(sentences):
    return [tokenizer.tokenize(sentence) for sentence in tqdm(sentences, desc="Tokenizing")]

real_data_tokenized = tokenize_sentences(real_data)
predicted_data_tokenized = tokenize_sentences(predicted_data)


def extract_ngrams(data, n):
    return [ng for sentence in tqdm(data, desc=f"Extracting {n}-grams") for ng in ngrams(sentence, n)]

all_ngrams = []
for n in range(1, 5):
    all_ngrams.extend(extract_ngrams(real_data_tokenized + predicted_data_tokenized, n))



k = 500
frequencies = Counter(all_ngrams)


min_length = 2
trivially_shared_ngrams = {
    str(ngram): count for ngram, count in frequencies.most_common(k)
    if len(ngram) >= min_length
}

references = [[sentence] for sentence in real_data_tokenized]
candidates = predicted_data_tokenized


print("Calculating CrystalBLEU Score...")


crystalBLEU_score = corpus_bleu(
    list_of_references=references,
    hypotheses=candidates,
    ignoring=trivially_shared_ngrams
)

print(f"CrystalBLEU Score: {crystalBLEU_score}")
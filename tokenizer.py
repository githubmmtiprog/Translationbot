# download_data.py
from datasets import load_dataset

dataset = load_dataset("Hoshikuzu/JParaCrawl", split='train[:1%]')

with open("ja.txt", "w", encoding="utf-8") as f_ja, open("en.txt", "w", encoding="utf-8") as f_en:
    for item in dataset:
        f_ja.write(item['translation']['ja'] + "\n")
        f_en.write(item['translation']['en'] + "\n")


print("Saved ja.txt and en.txt.")

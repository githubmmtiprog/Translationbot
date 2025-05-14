from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample sentence pairs
input_texts = ["hello", "how are you", "i am fine"]
target_texts = ["<start> hi <end>", "<start> i am good <end>", "<start> glad to hear <end>"]

# 1. Fit tokenizers
input_tokenizer = Tokenizer(oov_token="<OOV>")
target_tokenizer = Tokenizer(oov_token="<OOV>")
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

# 2. Convert text to sequences
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# 3. Pad sequences
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

encoder_input = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
decoder_input = pad_sequences([seq[:-1] for seq in target_sequences], maxlen=max_target_len - 1, padding='post')
decoder_target = pad_sequences([seq[1:] for seq in target_sequences], maxlen=max_target_len - 1, padding='post')
decoder_target = np.expand_dims(decoder_target, -1)

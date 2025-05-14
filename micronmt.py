import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Input/output sentence pairs (English to Japanese)
import urllib.request
import zipfile
import os

dataset_url = "https://github.com/makcedward/nlpaug/blob/master/example/jpn-eng.zip?raw=true"
dataset_path = "jpn-eng.zip"
extracted_file = "jpn.txt"

if not os.path.exists(extracted_file):
    print("Downloading dataset...")

    req = urllib.request.Request(
        dataset_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(req) as response:
        with open(dataset_path, 'wb') as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("Dataset extracted.")


# Read and preprocess the data
input_texts = []
target_texts = []

with open(extracted_file, encoding="utf-8") as f:
    for line in f:
        eng, jpn, _ = line.strip().split('\t')
        input_texts.append(eng.lower())
        target_texts.append(f"<start> {jpn} <end>")

# (Optional) Subsample for speed during testing
input_texts = input_texts[:10000]
target_texts = target_texts[:10000]


# Tokenizers for both languages
input_tokenizer = Tokenizer(oov_token="<OOV>", filters='')
target_tokenizer = Tokenizer(oov_token="<OOV>", filters='')

input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

# Convert texts to sequences
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Pad sequences
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

encoder_input = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
decoder_input = pad_sequences([seq[:-1] for seq in target_sequences], maxlen=max_target_len - 1, padding='post')
decoder_target = pad_sequences([seq[1:] for seq in target_sequences], maxlen=max_target_len - 1, padding='post')
decoder_target = np.expand_dims(decoder_target, -1)

# Vocabulary sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1


def build_seq2seq_model(input_vocab_size, target_vocab_size, input_len, target_len, embedding_dim=64, latent_dim=256):
    # Encoder
    encoder_inputs = layers.Input(shape=(input_len,), name="encoder_inputs")
    enc_emb = layers.Embedding(input_vocab_size, embedding_dim, name="encoder_embedding")(encoder_inputs)
    encoder_lstm = layers.LSTM(latent_dim, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = layers.Input(shape=(target_len,), name="decoder_inputs")
    dec_emb_layer = layers.Embedding(target_vocab_size, embedding_dim, name="decoder_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = layers.TimeDistributed(layers.Dense(target_vocab_size, activation='softmax'), name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Save encoder and decoder pieces
    encoder_model_inputs = encoder_inputs
    encoder_model_outputs = encoder_states

    decoder_state_input_h = layers.Input(shape=(latent_dim,))
    decoder_state_input_c = layers.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_lstm_outputs, state_h, state_c = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs_inf = decoder_dense(decoder_lstm_outputs)

    encoder_model = models.Model(encoder_model_inputs, encoder_model_outputs)
    decoder_model = models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs_inf] + decoder_states
    )

    return model, encoder_model, decoder_model


# Build the training model
# Build the models
model, encoder_model, decoder_model = build_seq2seq_model(
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    input_len=max_input_len,
    target_len=max_target_len - 1
)


# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit([encoder_input, decoder_input], decoder_target, epochs=100, batch_size=64)

# === Inference Setup ===



# Decoding function
def decode_sequence(input_seq, max_target_len, start_token, end_token):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.array([[target_tokenizer.word_index[start_token]]])
    decoded_sentence = []

    for _ in range(max_target_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, "")

        if sampled_word == end_token:
            break

        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return ' '.join(decoded_sentence)


# === Test the model with new inputs ===
new_inputs = ["Hi", "good evening"]
for sentence in new_inputs:
    seq = input_tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen=max_input_len, padding='post')
    translated = decode_sequence(padded_seq, max_target_len, start_token="<start>", end_token="<end>")
    print(f"Input: {sentence}\nGenerated: {translated}\n")

    import gradio as gr

def translate_input(text):
    seq = input_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_input_len, padding='post')
    result = decode_sequence(padded, max_target_len, start_token="<start>", end_token="<end>")
    return result

gr.Interface(fn=translate_input,
             inputs=gr.Textbox(lines=2, label="English Input"),
             outputs=gr.Textbox(label="Japanese Output"),
             title="Mini NMT: English to Japanese").launch(share=True)


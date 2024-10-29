# Example of Transformer.
# https://www.tensorflow.org/text/tutorials/transformer
# https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb
# https://keras.io/examples/timeseries/timeseries_transformer_classification/  Timeseries classification. 

# Translation from  Portuguese to English.

# Steps for the transformer.
# 1. Tokenizer to assign ID to words.
# 2. Embedding to convert each words to a vector.
# 3. Use Attention layer(instead of LSTM) to fit the input data to target data. 
#    LSTM or RNN has problem with long sequence data.  Attention layer has not problem with it 
#     because it takes the whole sequence of data into the attention layer all at once, instead
#     of one by one (LSTM).
# 4. Use encorder/decorder to finally output the result.  

import os
import numpy as np
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text  # incase of errors.  pip upgrade tensorflow_text (also upgrade tensorflow)
import keras.utils

os.chdir('C:/my_working_env/deeplearning_practice/transformer/')

#------------------------------------------------------------------------------------------------------------------

# Download dataset.  https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

# Note the type is "tensorflow Dataset". https://www.tensorflow.org/api_docs/python/tf/data/Dataset
train_examples = examples['train']
val_examples = examples['validation']

#----- Display example sentences -------------------------------
# note train_examples.batch().take(1) returns two elements pt and en.  
# To visualize tf dataset inside, run " train_examples.batch(1).take(1).__iter__().get_next()[0] "  https://stackoverflow.com/questions/56820723/what-is-tensorflow-python-data-ops-dataset-ops-optionsdataset
#  another way is to use .batch(1).take(1) with for loop like below.
for pt_examples, en_examples in train_examples.batch(5).take(1): # display the first 5 sentences. take(1) means only take the first element.
    print(); print('> Examples in Portuguese:')
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'));

    print(); print('> Examples in English:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'));
    print()
#----------------------------------------------------------------

# Use the existing Tokenizer model file.  Tokenizer converts each word to an ID number.

# download the saved model. https://www.tensorflow.org/text/guide/subwords_tokenizer
# Once you download, you don't need them.
model_name = 'ted_hrlr_translate_pt_en_converter'
#tf.keras.utils.get_file(
#    f'{model_name}.zip',
#    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
#    cache_dir='.', cache_subdir='', extract=True  )

# Load the downloaded file.
tokenizers = tf.saved_model.load(model_name)

# Remove string that starts with "__xxxx__".  
[item for item in dir(tokenizers.en) if not item.startswith('_')]

# Visualize the the tokenized sentence.
encoded = tokenizers.en.tokenize(en_examples) # Note only 5 sentences(batch).  see above.
print();  print('> This is a padded-batch of token IDs:')
for row in encoded.to_list():
    print(row)
print()

# Detokenize the word.
round_trip = tokenizers.en.detokenize(encoded)
print('> This is human-readable text:')
for line in round_trip.numpy():
    print(line.decode('utf-8'))

print('> This is the text split into tokens:')
# The output demonstrates the b"subword" (b stands for bytes) aspect of the tokenization.
# e.g. word "searchability is decomposed into "search" and "##ability".  
# Note both pt and en both has [START] and [END] elements.  These are needed.
tokens = tokenizers.en.lookup(encoded)
print(tokens); print()

#--------- Prepare the input, output data ---------------------------------
# might be better to use keras preprocessing.

def prepare_batch( pt, en ):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]      # Trim to MAX_TOKENS.
    pt = pt.to_tensor()    # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    # (pt, en_input) is the input, and (en_label)=target.  Transformer is made this way.
    return (pt, en_inputs), en_labels  
    
def make_batches(ds):
    # note xxx.batch() produce two elements pt and en.  Use AUTOTUNE for efficiency. 
    return (
          ds
          .shuffle(BUFFER_SIZE)  # it shuffles the data x times (x=buffer_size). e.g. the first sentense becomes nth sentence.
          .batch(BATCH_SIZE)     # take the first x sentences.  x(batch_size)
          .map( prepare_batch )   #  tf.data.AUTOTUNE)  # use map function https://www.gcptutorials.com/article/how-to-use-map-function-with-tensorflow-datasets  
          .prefetch(buffer_size=tf.data.AUTOTUNE)
          )

MAX_TOKENS=128   # if max_token is smaller bigger than max words of sentence, then max_token=max word.
BUFFER_SIZE = 20000
BATCH_SIZE = 64

# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

# visualize the dataset. The en and en_labels are the same, just shifted by 1
print()
for (pt, en), en_labels in train_batches.take(1): # take one batch of training.
    print(pt)  # dimension=batch x max_token
    print(en)
    print(en_labels)
    print(en[0][:10])
    print(en_labels[0][:10])    
    break

#------------ Positional encoding layer --------------------------------------
# LSTM can recognize the order of the sequence, but transformer need positional
# layer to identify the sequence.  
# positional encoding function is just sine cosine function.

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]     # make it a column vector [length x 1] dimension.  np.reshape(length,1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # this is a row vector [depth x 1] dimension.

    angle_rates = 1 / (10000**depths)         # dimension = [1 x depth]
    angle_rads = positions * angle_rates      # dimension = [pos, depth]

    pos_encoding = np.concatenate(
                                  [np.sin(angle_rads), np.cos(angle_rads)],
                                  axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(length=2048, depth=512)

# Check the shape.
print(pos_encoding.shape)  # 2048 x 512

# plot 2D
plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
plt.ylabel('Depth'); plt.xlabel('Position')
plt.colorbar()
plt.show(); plt.close()  

# plot 1D graph along position=1000
pos_encoding/=tf.norm(pos_encoding, axis=1, keepdims=True)
px = pos_encoding[1000,:]

dots = tf.einsum('pd,d -> p', pos_encoding, px)
plt.subplot(2,1,1)
plt.plot(dots)
plt.ylim([0,1])
plt.plot([950, 950, float('nan'), 1050, 1050],  # this is just drawing vertical lines.
         [0,1,float('nan'),0,1], color='k', label='Zoom')
plt.legend()
plt.subplot(2,1,2)
plt.plot(dots)
plt.xlim([950, 1050])
plt.ylim([0,1])
plt.show(); plt.close()

# Now create the positional layer.
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):  # d_model=embedding dimension.
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding( vocab_size, d_model, mask_zero=True )  # Remember that empty words are padded with zeros in the sentence (to make sequence length fixed), that zero should be included or not.
        self.pos_encoding = positional_encoding( length=2048, depth=d_model )

    def compute_mask(self, *args, **kwargs):  # this is needed in embedding layer to assign 1(there is word) or 0(no word, padded with zero).
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x *= tf.math.sqrt( tf.cast(self.d_model, tf.float32) )
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

embed_pt = PositionalEmbedding( vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
embed_en = PositionalEmbedding( vocab_size=tokenizers.en.get_vocab_size(), d_model=512)
pt_emb = embed_pt(pt)
en_emb = embed_en(en)  # dimension = batch x max_words(in sentence) x embedding_size

print( en_emb._keras_mask )  # dimension = batch x max_words(in sentence). True means there is a word.

#--------------- Now for the attention layer ----------------------------------------------

# Attention layer tries to find the best or close match of the inputs to the outputs words.
# It has three layers.  CrossAttention layer, GlobalSelfAttention layer, CausalSelfAttention layer.
# BaseAttention is just setting up layers.
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()  # Normalized the vector of the whole inputs, not the batch of inputs vector. 
        self.add = tf.keras.layers.Add()

# CrossAttention layer connects the encorder and decorder.
class CrossAttention(BaseAttention):  # Note the BaseAttention (you don't need tf.keras.layers.Layer)
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

# Note that dimension is the same as embedded vector en_emb or pt_emb.
sample_ca = CrossAttention(num_heads=2, key_dim=512)
print(sample_ca(en_emb, pt_emb).shape)

# GrossSelfAttention layer is the initial layer of the inputs(pt).  It is kind of like RNN & CNN combined.
# It is able to process sequential data(like RNN) in parallel, and able to detect features(like CNN) with one layer.
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)

# Note that dimension is the same as embedded vector en_emb or pt_emb.
print(sample_gsa(pt_emb).shape)

# CasualSelfAttention layer is the initial layer of the other inputs(en).
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)

# Note that dimension is the same as embedded vector en_emb or pt_emb.
print(sample_csa(en_emb).shape)

out1 = sample_csa(embed_en(en[:, :3])) 
out2 = sample_csa(embed_en(en))[:, :3]
print(); print( tf.reduce_max(abs(out1 - out2)).numpy() )  # It's nearly zero.


#-------------- FeedForward layer -----------------------------------------------------

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units1, units2, dropout_rate=0.1):
        super().__init__()
        
        self.seq = tf.keras.Sequential([
          tf.keras.layers.Dense(units2, activation='relu'),
          tf.keras.layers.Dense(units1),
          tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x

sample_ffn = FeedForward(512, 2048)  # Note 512 is same as the embedding size.

# The output has the same dimension.
print(); print(en_emb.shape)
print(sample_ffn(en_emb).shape)

#------------ Encorder layer ---------------------------------------
# stack of GlobalAttention layer and feedforward layer.
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)

print(); print(pt_emb.shape)
print(sample_encoder_layer(pt_emb).shape)

#---------- Encorder ------------------------------------
# encorder layer plus the embedding layer.

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model )

        self.enc_layers = [
                        EncoderLayer(d_model=d_model, num_heads=num_heads,
                                     dff=dff, dropout_rate=dropout_rate)
                        for _ in range(num_layers)
                        ]
            
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.

# Instantiate the encoder.
sample_encoder = Encoder(num_layers=4,
                         d_model=512,
                         num_heads=8,
                         dff=2048,
                         vocab_size=8500)

sample_encoder_output = sample_encoder(pt, training=False)

# Print the shape.
print(pt.shape)
print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.

#--------------- Decorder layer ------------------------------------------------------
# Casual attention and Cross attention layers.
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x

sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)

sample_decoder_layer_output = sample_decoder_layer(
    x=en_emb, context=pt_emb)

print(en_emb.shape)
print(pt_emb.shape)
print(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)`

#----------------- Decorder -------------------------------------------------
# Decorder layer, positional encording, and embedding layers.

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
          x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x

# test the decoder.
sample_decoder = Decoder(num_layers=4,
                         d_model=512,
                         num_heads=8,
                         dff=2048,
                         vocab_size=8000)

output = sample_decoder(
    x=en,
    context=pt_emb)

# Print the shapes.
print(en.shape)
print(pt_emb.shape)
print(output.shape)

#------------ Transformer ---------------------------------------------------
class Transformer(tf.keras.Model):  # Note if tf.keras.Model use, you don't need to call tf.keras.Model(inputs=x, outputx=y).
    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        context, x  = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits

#=================== Check transformer output ===========================================

num_layers = 4
d_model = 128  # number of embedding size.
dff = 512      # number of units in the dense layer of feed forward layers.
num_heads = 8  # number of heads in GlobalSelfAttention layer.
dropout_rate = 0.1

import pdb; pdb.set_trace()  

transformer = Transformer(
                        num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
                        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
                        dropout_rate=dropout_rate)

output = transformer((pt, en))

print(); print(en.shape)  # dimension = batch_size x max_length_sentence
print(pt.shape)           # dimension = batch_size x max_length_sentence
print(output.shape)       # dimension = batch_size x max_length_sentence x #_of_vocabs(english).

# MultiHeadAttention layer produce scores.  Importance of output words (decorder) given the input words.
attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores  
print(); print(attn_scores.shape)  # (batch, num_heads, max_target_sentence_length, max_input_sentence_length )

#================ Train with Transformer =============================

# Vary learning rate.
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# Plot the learning rate vs. epochs.
plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')
plt.show(); plt.close()

# Define loss.
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

# Define accuracy.
def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

transformer.compile(
                loss=masked_loss,
                optimizer=optimizer,
                metrics=[masked_accuracy]
                )

transformer.fit(train_batches,
                epochs=1,  # default=20
                validation_data=val_batches)

#------------- Prediction ----------------------------------
class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # As the output language is English, initialize the output with the
    # English `[START]` token.
    start_end = self.tokenizers.en.tokenize([''])[0]  # Set english decoder inputs as empty string.
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

    tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    return text, tokens, attention_weights

translator = Translator(tokenizers, transformer)


def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')
  
sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

import pdb; pdb.set_trace()  




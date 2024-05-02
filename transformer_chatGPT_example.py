# Example of Encorder-Decorder from chatGPT.


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Masking
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization


class EncoderDecoderTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1):
        super(EncoderDecoderTransformer, self).__init__()
        self.encoder_pos_encoding = positional_encoding(pe_input, d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.encoder_dropout = Dropout(rate)

        self.decoder_pos_encoding = positional_encoding(pe_target, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.decoder_dropout = Dropout(rate)

        self.final_layer = Dense(1)  # Output layer for t+1 prediction

    def call(self, inputs, training=True):
        enc_input, dec_input = inputs

        enc_output = self.encode(enc_input, training=training)
        dec_output = self.decode(dec_input, enc_output, training=training)
        final_output = self.final_layer(dec_output)

        return final_output

    def encode(self, inputs, training=True):
        x = inputs
        x += self.encoder_pos_encoding[:, :tf.shape(x)[1], :]
        x = self.encoder_dropout(x, training=training)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)

        return x

    def decode(self, inputs, enc_output, training=True):
        x = inputs
        # x *= tf.math.sqrt(tf.cast(x.shape[-1], tf.float32))  You may need this factor if the number of feature is more than 2.  
        x += self.decoder_pos_encoding[:, :tf.shape(x)[1], :]
        x = self.decoder_dropout(x, training=training)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, training=training)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=d_model)
        self.dropout1 = Dropout(rate)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.dense = Dense(units=dff, activation='relu')
        self.dropout2 = Dropout(rate)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=True):
        attn_output = self.multihead_attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)
        dense_output = self.dense(out1)
        dense_output = self.dropout2(dense_output, training=training)
        out2 = self.layer_norm2(out1 + dense_output)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.multihead_attention1 = MultiHeadAttention(num_heads=num_heads,
                                                       key_dim=d_model)
        self.multihead_attention2 = MultiHeadAttention(num_heads=num_heads,
                                                       key_dim=d_model)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = LayerNormalization(epsilon=1e-6)
        self.dense = Dense(units=dff, activation='relu')

    def call(self, inputs, enc_output, training=True):
        attn1 = self.multihead_attention1(inputs, inputs)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layer_norm1(inputs + attn1)

        attn2 = self.multihead_attention2(out1, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layer_norm2(out1 + attn2)

        dense_output = self.dense(out2)
        dense_output = self.dropout3(dense_output, training=training)
        out3 = self.layer_norm3(out2 + dense_output)

        return out3


def positional_encoding(position, d_model):
    angle_rads = get_angles(tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                            d_model)
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(position, i, d_model):
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angle_rates


# Example usage:
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
pe_input = 1000
pe_target = 1000

# Create an instance of the Encoder-Decoder Transformer model
transformer = EncoderDecoderTransformer(num_layers, d_model, num_heads, dff,
                                        pe_input, pe_target)

# Prepare encoder input and decoder input for a single prediction
enc_input = tf.random.uniform((1, sequence_length), dtype=tf.float32, minval=0, maxval=100)
dec_input = tf.random.uniform((1, sequence_length), dtype=tf.float32, minval=0, maxval=100)

# Call the model to get the predicted output for t+1
prediction = transformer([enc_input, dec_input])

# Print the shape of the prediction.  the shape of the prediction will be (1, 1).
print(prediction.shape)  


#---------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Masking
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization


class EncoderDecoderTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1):
        super(EncoderDecoderTransformer, self).__init__()
        self.encoder_pos_encoding = positional_encoding(pe_input, d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.encoder_dropout = Dropout(rate)

        self.decoder_pos_encoding = positional_encoding(pe_target, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.decoder_dropout = Dropout(rate)

        self.final_layer = Dense(1)  # Output layer for t+1 prediction

    def call(self, inputs, training=True):
        enc_input, dec_input = inputs

        enc_output = self.encode(enc_input, training=training)
        dec_output = self.decode(dec_input, enc_output, training=training)
        final_output = self.final_layer(dec_output)

        return final_output

    def encode(self, inputs, training=True):
        x = inputs
        x = Masking()(x)  # Masking layer to mask padded zeros
        x += self.encoder_pos_encoding[:, :tf.shape(x)[1], :]
        x = self.encoder_dropout(x, training=training)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)

        return x

    def decode(self, inputs, enc_output, training=True):
        x = inputs
        x += self.decoder_pos_encoding[:, :tf.shape(x)[1], :]
        x = self.decoder_dropout(x, training=training)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, training=training)

        return x


#-------------------------------------------------------------------------------------

# If you have missing values in the encoder inputs and want to pad them with zeros while masking them so that they don't affect the prediction, you can use the Masking layer in TensorFlow. 
# The Masking layer masks the input sequences by setting a mask value (mask_value) for the padded elements, indicating that those elements should be ignored during computation.

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Masking
from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization


class EncoderDecoderTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, rate=0.1):
        super(EncoderDecoderTransformer, self).__init__()
        self.encoder_pos_encoding = positional_encoding(pe_input, d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.encoder_dropout = Dropout(rate)

        self.decoder_pos_encoding = positional_encoding(pe_target, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]
        self.decoder_dropout = Dropout(rate)

        self.final_layer = Dense(1)  # Output layer for t+1 prediction

    def call(self, inputs, training=True):
        enc_input, dec_input = inputs

        enc_output = self.encode(enc_input, training=training)
        dec_output = self.decode(dec_input, enc_output, training=training)
        final_output = self.final_layer(dec_output)

        return final_output

    def encode(self, inputs, training=True):
        x = inputs
        x = Masking()(x)  # Masking layer to mask padded zeros
        x += self.encoder_pos_encoding[:, :tf.shape(x)[1], :]
        x = self.encoder_dropout(x, training=training)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)

        return x

    def decode(self, inputs, enc_output, training=True):
        x = inputs
        x += self.decoder_pos_encoding[:, :tf.shape(x)[1], :]
        x = self.decoder_dropout(x, training=training)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, training=training)

        return x

# If input data is masked, you'll need to modigy the loss function.

def masked_mse_loss(y_true, y_pred, mask):
    # Apply the mask to the predictions and targets
    masked_pred = tf.boolean_mask(y_pred, mask)
    masked_true = tf.boolean_mask(y_true, mask)
    
    # Compute the mean squared error loss only for unmasked elements
    mse_loss = tf.reduce_mean(tf.square(masked_pred - masked_true))
    return mse_loss

model.compile(optimizer='adam', loss=masked_mse_loss)



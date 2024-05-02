# Transformer time-series with 2 kinds of positional encoding.
# Input has 2 features.  No input data. 
# Note this is just mock code.  There is no training.  

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def positional_encoding2(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]     # make it a column vector [length x 1] dimension.  np.reshape(length,1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # this is a row vector [depth x 1] dimension.

    angle_rates = 1 / (10000**depths)         # dimension = [1 x depth]
    angle_rads = positions * angle_rates      # dimension = [pos, depth]

    pos_encoding = np.concatenate(
                                  [np.sin(angle_rads), np.cos(angle_rads)],
                                  axis=-1) 
                                  
    pos_encoding = pos_encoding.reshape(-1, pos_encoding.shape[0], pos_encoding.shape[1] ) # reshape to 1x10x2. You could also use np.newaxis.  
    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos, i, d_model):
    #angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rates = 1 / tf.pow(10000.0, (2 * ( tf.cast(i, tf.float32) // 2)) / tf.cast(d_model, tf.float32))
    return tf.cast(pos, tf.float32) * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        tf.range(position)[:, tf.newaxis],  # newaxis add dimension (increase tensor rank)
        tf.range(d_model)[tf.newaxis, :],   # tf.range is basically the same as np.arange()  
        d_model
    )

    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)  # note axis=-1 and axis=1 are same except 1d vector.  
    pos_encoding = pos_encoding[tf.newaxis, ...]  # reshape to 1x10x2 from 10x2.

    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_model():
    
    lookup_time = 10
    num_features = 2
    input_shape = (lookup_time, num_features)  # 10=lookup_size, 2=# of columns.  
    d_model = 16
    num_heads = 3 
    num_encoder_layers = 2
    
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Positional encoding layer
    position = input_shape[0]    # Integer.  Lookup window size.
    pos_encoding = positional_encoding(position, num_features)  # 2=number of features.
    x = layers.Add()([inputs, pos_encoding[:, :input_shape[0], :]])

    # Multi-head self-attention layers
    for i in range(num_encoder_layers):
        x = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        attn_output = attn(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)
        ff = layers.Dense(4*d_model, activation='relu')
        ff_output = ff(x)
        ff_output = layers.Dense(num_features)(ff_output)  # 2=# of features.
        x = layers.Add()([x, ff_output])

    # Final dense layer for prediction. 
    output = layers.Dense(1)(x[:, -1, :])  # Need to reduce tensor rank to 2 from 3.  

    # Define and compile model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')

    return model
    
model = transformer_model()
import pdb; pdb.set_trace()  









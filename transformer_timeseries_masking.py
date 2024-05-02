
# Example of timeseries Transformer. 
# https://keras.io/examples/timeseries/timeseries_transformer_classification/ 
# Transformer model with 1 features, with positional encoding, with masking.  No decoder.  
# Note the positional encoder can accept any dimension (usually dimension needs to be even number).

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

T = 10    # send T past values, and predict the next value
D = 1   # number of input column

os.chdir('C:\my_working_env\deeplearning_practice\\transformer')


#------------- Preparing data NxTxD ---------------------------------------------

num_points = 1000

# Sample training data
encoder_inputs = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
target_data = [12, 22, 33, 44]

num_total = 1000
num_data = int(num_total/T)
encoder_inputs = np.arange(1, num_total+1 ).reshape( num_data, T, D )/num_total
target_data = np.arange(T+1, num_total+1+T, T  )/num_total

# padded with zeros.
for n in range( len(encoder_inputs) ):
    encoder_inputs[n, 9, :] = 0.0  

# Extract arrays from random indices.
rand_ind = random.sample( range(num_data), k=int(num_data*0.8)  )
non_rand_ind = list(set([*range(0,num_data)]) - set(rand_ind))

encoder_train = encoder_inputs[rand_ind]  
target_train = target_data[rand_ind]

encoder_test = encoder_inputs[non_rand_ind]  
target_test = target_data[non_rand_ind]

# Convert input and target data to tensor arrays.
encoder_train = tf.convert_to_tensor(encoder_train, dtype=tf.float32)
target_train = tf.convert_to_tensor(target_train, dtype=tf.float32)

encoder_test = tf.convert_to_tensor(encoder_test, dtype=tf.float32)
target_test = tf.convert_to_tensor(target_test, dtype=tf.float32)


#----------------- Setup neural network. -------------------------------------
        
class transformer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        head_size = 64
        num_heads = 4
        num_filter = 4
        dropout = 0.1
                
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.mha = tf.keras.layers.MultiHeadAttention(
                                                    key_dim=head_size, 
                                                    num_heads=num_heads, 
                                                    dropout=dropout,
                                                    )

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv1D = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=1, activation="relu",)
        self.conv1D_2 = tf.keras.layers.Conv1D(filters=D, kernel_size=1) 
        
    def call(self, x):
        inputs = x
        x = self.mha(x,x)
        x = self.dropout(x)
        res = x + inputs

        x = self.conv1D(res)
        x = self.dropout(x)
        x = self.conv1D_2(x)
        return x + res
        
        
class transformer_model(tf.keras.Model):
    def __init__(self, T, D):
        super().__init__()
        
        self.num_transformer = 2
        dense_units = 64
        dropout = 0.0
        
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")
        
        self.dense_layer = tf.keras.layers.Dense(dense_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.transformer_layer = transformer()
        self.output_layer = tf.keras.layers.Dense(D) 
        self.pos_encoding = self.positional_encoding( T, D )

    def positional_encoding(self, length, d_model):
        positions = np.arange(length)[:, np.newaxis]
        angles = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
        positional_encodings = positions * angles

        positional_encodings[:, 0::2] = np.sin(positional_encodings[:, 0::2])
        positional_encodings[:, 1::2] = np.cos(positional_encodings[:, 1::2])
        return tf.cast(positional_encodings[ np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        
        # inputs needs to be NxTxD dimension.
        # Add masking layer to ignore padded 0.0
        x = tf.keras.layers.Masking(mask_value=0.0)(x)
        
        # Note the positional encoding dimension needs to be the same as inputs dimension (NxTxD).
        # The current positional encoding can be any dimension, but other positional encoding can only accept even number of dimension (D should be even).
        x = tf.keras.layers.Add()([x, self.pos_encoding])  
        
        for _ in range(self.num_transformer):  
            x = self.transformer_layer(x)
        
        x = self.pooling(x)  # Reduce dimension from NxTxD to NxT.  
        x = self.dense_layer(x)
        x = self.dropout(x)
        outputs = self.output_layer(x)
        
        return outputs

def masked_mse(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(tf.square(y_true - y_pred) * mask) / tf.reduce_sum(mask)

#--------- Train the model ----------------------------------------------

model = transformer_model( T, D )  

model.compile(
            loss=masked_mse,    #tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam( learning_rate=0.001 ),
            #metrics=["sparse_categorical_accuracy"],
            )

rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  #"val_loss", 
                                            patience=10, verbose=1)
es = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

# test with input data.
#dummy_x = np.random.rand( 1, T, D ) 
#model( dummy_x ) 

res = model.fit(
                encoder_train,
                target_train,
                validation_split=0.1,
                epochs=1000,
                batch_size=10,
                callbacks=[ rlr, es ] ,
                )
        

#model.predict( encoder_test[0:2] )
#model.evaluate( encoder_test, encoder_test, verbose=1)

plt.plot( res.history['loss'], label='loss' ) 
plt.plot( res.history['val_loss'], label='val_loss' ) 
plt.legend()
plt.xlabel('epoch')
plt.show()


print()
for n in range(len(target_train)):
    print("prediction = ",  model.predict( encoder_train[n][np.newaxis,:,:], verbose=False)[0,0], 
    " Target = ",  target_train[n].numpy() )
print("Transformer MSE*1000(train)= ", model.evaluate( encoder_train, target_train )*1000 )  

print()
for n in range(len(target_test)):
    print("prediction = ",  model.predict(encoder_test[n][tf.newaxis,:,:], verbose=False)[0,0], 
    " Target = ",  target_test[n].numpy() )
print("Transformer MSE*1000(test)= ", model.evaluate( encoder_train, target_train )*1000 )  


import pdb; pdb.set_trace()  




    


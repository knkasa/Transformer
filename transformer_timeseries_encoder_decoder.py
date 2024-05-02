# Example of timeseries Transformer. 
# https://keras.io/examples/timeseries/timeseries_transformer_classification/ 
# Transformer model with 1 features, with positional encoding and decoder.
# https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
# Note the positional encoding can only be even in this code, which is why it is commented out.  

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

T = 10  # sequence length.
D = 1   # number of feature.  

os.chdir('C:\my_working_env\deeplearning_practice\\transformer')

#----------------- Setup neural network. -------------------------------------
        
class positional_encoding_class():
    def __init__(self):
        
        self.pos_encoding = self.positional_encoding( T, D )
        
    def positional_encoding(self, length, depth):
        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]     # make it a column vector [length x 1] dimension.  np.reshape(length,1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # this is a row vector [depth x 1] dimension.

        angle_rates = 1 / (10000**depths)         # dimension = [1 x depth]
        angle_rads = positions * angle_rates      # dimension = [pos, depth]

        pos_encoding = np.concatenate( [np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
        pos_encoding = pos_encoding.reshape(-1, pos_encoding.shape[0], pos_encoding.shape[1] ) # reshape to 1xTx2. You could also use np.newaxis.  
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
class encoder_transformer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        head_size = 128
        num_heads = 4
        num_filter = 4
        dropout = 0.0
                
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Note attention layer requires dimension to be NxTxD just like LSTM.
        self.mha = tf.keras.layers.MultiHeadAttention(
                                                    key_dim=head_size, 
                                                    num_heads=num_heads, 
                                                    dropout=dropout,
                                                    )

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv1D = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=1, activation="relu",)    # dimension = NxTxF (F=# of filters)
        self.conv1D_2 = tf.keras.layers.Conv1D(filters=D, kernel_size=1)    # we need dimension=NxTxD which is why the filter is set to D.
        
    def call(self, inputs):
        '''
        x = inputs
        #x = self.layer_norm(x)
        x = self.mha(x,x)
        x = self.dropout(x)
        res = x + inputs
        
        #x = self.layer_norm(res)
        x = self.conv1D(res)  # Note conv1D has dimension NxTxD. conv2D has dimension NxT1xT2xD. 
        x = self.dropout(x)
        x = self.conv1D_2(x)
        return x + res
        '''
        #'''
        x = inputs
        x = self.mha(x, x)  # attention layer
        x = self.dropout(x)
        x = x+inputs  #self.layer_norm(x+inputs)  # add & norm
        #x = self.layer_norm(x+inputs)
        res = x

        x = self.conv1D(x)    # Note conv1D has dimension NxTxD. conv2D has dimension NxT1xT2xD. 
        x = self.dropout(x)
        x = self.conv1D_2(x)  # This is needed to correct the dimension.
        x = x+res  #self.layer_norm(x+res)  # add & norm
        #x = self.layer_norm(x+res)
        return x
        #'''

        
class decoder_transformer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        head_size = 128
        num_heads = 4
        num_filter = 4
        dropout = 0.0
        
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Note attention layer requires dimension to be NxTxD just like LSTM.
        self.mha = tf.keras.layers.MultiHeadAttention(
                                                    key_dim=head_size, 
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    )
        
        self.encoder_layer = encoder_transformer()
        
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.conv1D = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=1, activation="relu")    # dimension = NxTxF (F=# of filters)
        self.conv1D_2 = tf.keras.layers.Conv1D(filters=D, kernel_size=1)    # we need dimension=NxTxD which is why the filter is set to D.
        
    def call(self, encoder_inputs, decoder_inputs):
        '''
        x = decoder_inputs
        #x = self.layer_norm(x)
        x = self.mha(x,x)
        x = self.dropout(x)
        res = x + decoder_inputs
        
        #x = self.layer_norm(res)
        x = self.mha( res, encoder_inputs ) 
        x = self.dropout(x)
        res = res + x
        
        #x = self.layer_norm(res)
        x = self.conv1D(res)
        x = self.dropout(x)
        x = self.conv1D_2(x)
        return x
        '''
        #'''
        x0 = decoder_inputs
        x = self.mha(x0, x0)
        x = self.dropout(x)
        x = x+x0  #self.layer_norm(x+x0)
        #x = self.layer_norm(x+x0)
        
        res = x
        x = self.mha(x, encoder_inputs)  # attention layer with decoder
        x = self.dropout(x)
        x = res+x  #self.layer_norm(res + x)
        #x = self.layer_norm(res + x)
        
        res = x
        x = self.conv1D(x)    # Note conv1D has dimension NxTxD. conv2D has dimension NxT1xT2xD. 
        x = self.dropout(x)
        x = self.conv1D_2(x)
        x = res+x  #self.layer_norm(res+x)
        #x = self.layer_norm(res+x)  
        return x
        #'''
        

class transformer_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.num_block_layers = 4
        self.num_encoder = 2
        self.num_decoder = 2
        dense_units = 64
        dropout = 0.0
        
        # x has dimension NxTxD.
        # GlobalAveragePooling1D will take the average of T dimension, and reduce array dimension to NxD.
        # "channels_first" is to reduce dimension to NxT.  "chaanels_last" is to reduce dimension to NxD.
        # We need to reduce the dimension because dense layer only accepts tensor of rank 2.
        # If you have multiple columns D, then you need to flatten T & D to make tensor rank 2.  NxTxD -> Nx(T*D) dimension.  
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")
        
        self.dense_layer = tf.keras.layers.Dense(dense_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.output_layer = tf.keras.layers.Dense(D)
        self.dense_layer2 = tf.keras.layers.Dense(dense_units)
        self.add_layer = tf.keras.layers.Add()
        
        self.positional_encoding = positional_encoding_class()
        self.encoder_layer = encoder_transformer()
        self.decoder_layer = decoder_transformer()

    def call(self, inputs, training):
    
        encoder_inputs, decoder_inputs = inputs
        
        # Bug: Note that positional_encoding has dimension NxTx2, not NxTxD.
        #x = self.add_layer([encoder_inputs, self.positional_encoding.pos_encoding])
        x = encoder_inputs
        
        # Note that after the transformer_layer, the input dimension stays the same. 
        for _ in range(self.num_encoder):  # stack number of transformer layers.
            x = self.encoder_layer(x)
        
        encoder_inputs = x
        
        #x = self.add_layer([decoder_inputs, self.positional_encoding.pos_encoding])
        x = decoder_inputs
        
        for _ in range(self.num_decoder):
            x = self.decoder_layer(encoder_inputs, x)
        
        # If the input data is 2D, no need for self.pooling. 
        #x = self.dense_layer2(x[:, -1, :])  #this is if D>1 case. [:, -1, :] is needed to reduce dimension from NxTxD to NxD. -1 means output produce t+1 prediction.
        x = self.pooling(x)  # Reduce dimension from NxTxD to NxT.  
        
        #x = self.dropout(x)
        outputs = self.output_layer(x)
        # note that -1 index is chosen to only predict t+1.
        # Another way is to flatten T & D to make tensor rank 2.  NxTxD -> Nx(T*D) dimension. 
        
        return outputs


#--------- Train the model ----------------------------------------------

model = transformer_model()  
#model.build( input_shape=(None, T, D) )
#exit()

model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam( learning_rate=0.001 ),
            #metrics=["sparse_categorical_accuracy"],
            )

rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",  #"val_loss", 
                                            patience=30, verbose=1)
es = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)


#------------- input data ----------------------------------------

# test
#x1 = np.random.rand( 1, T, D )
#x2 = np.random.rand( 1, T, D )
#x = ( x1, x2 ) 
#model( x ) 

# Sample training data
encoder_inputs = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
decoder_inputs = [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                  [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]
target_data = [12, 22, 33, 44]

num_total = 1000
num_data = int(num_total/T)
encoder_inputs = np.arange(1, num_total+1 ).reshape( num_data, T, D )/num_total
decoder_inputs = np.arange(T+1, num_total+1+T ).reshape( num_data, T, D )/num_total
target_data = np.arange(T+1, num_total+1+T, T  )/num_total

# Extract arrays from random indices.
rand_ind = random.sample( range(num_data), k=int(num_data*0.8)  )
non_rand_ind = list(set([*range(0,num_data)]) - set(rand_ind))

encoder_train = encoder_inputs[rand_ind]  
decoder_train = decoder_inputs[rand_ind]
target_train = target_data[rand_ind]

encoder_test = encoder_inputs[non_rand_ind]  
decoder_test = decoder_inputs[non_rand_ind]
target_test = target_data[non_rand_ind]

# Convert input and target data to tensor arrays.
encoder_train = tf.convert_to_tensor(encoder_train, dtype=tf.float32)
decoder_train = tf.convert_to_tensor(decoder_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(target_train, dtype=tf.float32)

encoder_test = tf.convert_to_tensor(encoder_test, dtype=tf.float32)
decoder_test = tf.convert_to_tensor(decoder_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(target_test, dtype=tf.float32)

x_train_tup = ( encoder_train, decoder_train ) 
x_test_tup = ( encoder_test, decoder_test )

res = model.fit(
        x_train_tup,
        y_train,
        validation_split=0.1,
        epochs=5000,
        batch_size=20,
        callbacks=[rlr, es],
        )
 
print()
for n in range(len(y_train)):
    print("prediction = ",  model.predict((encoder_train[n][tf.newaxis,:,:], decoder_train[n][tf.newaxis,:,:]), verbose=False)[0,0], 
            " Target = ",  y_train[n].numpy() )
print("Transformer MSE*1000(train)= ", model.evaluate( x_train_tup, y_train )*1000 )  

print()
for n in range(len(y_test)):
    print("prediction = ",  model.predict((encoder_test[n][tf.newaxis,:,:], decoder_test[n][tf.newaxis,:,:]), verbose=False)[0,0], 
            " Target = ",  y_test[n].numpy() )
print("Transformer MSE*1000(test)= ", model.evaluate( x_test_tup, y_test )*1000 )  


plt.plot( res.history['loss'], label='loss' ) 
plt.plot( res.history['val_loss'], label='val_loss' ) 
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Transformer')
plt.show()

import pdb; pdb.set_trace()  


 









    
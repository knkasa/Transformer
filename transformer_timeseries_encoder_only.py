# Example of timeseries Transformer. 
# https://keras.io/examples/timeseries/timeseries_transformer_classification/ 
# Transformer model with 2 features, with positional encoding.  No decoder.  
# Note positional encoding https://www.geeksforgeeks.org/positional-encoding-in-transformers/

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

T = 20    # send T past values, and predict the next value
D = 2   # number of input column

os.chdir('C:\my_working_env\deeplearning_practice\\transformer')


#------------- Preparing data ---------------------------------------------

num_points = 1000

time = np.linspace( 0, 50*np.pi, num_points ) 
out = np.sin( time )
out2 = np.cos( time )

plt.plot( time, out, 'o-', time, out2, 'o-' )
plt.show() 

#--------- Prepare sets of data (NxTxD) -------------------------------------


out = np.reshape( out, [-1,1] )   # column 1
out2 = np.reshape( out2, [-1,1] )   # column 2
out3 = np.concatenate( (out, out2), axis=1 )   # combine two columns

X=[]
Y=[]
for t in range(len(out) - T):
    x = out3[t:t+T]
    X.append(x)
    y = out3[t+T]
    Y.append(y)

X = np.array(X).reshape(-1, T, D)   # make it NxTxD
Y = np.array(Y)

N = len(X)

# Note X[0] x[1] ... are older input.  x[-1] is the newest input.
print(" Send X[0,:,:] = " )
print( X[0,:,:] )
print(" and predict the next value Y[0] = ")
print( Y[0] )  
print(" Expected result should be ")
print( out3[0:T+1,:]   );  print() 

x_train = X[0:N//2]
x_test = X[N//2:]

y_train = Y[0:N//2]
y_test = Y[N//2:]


#----------------- Setup neural network. -------------------------------------
        
class transformer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
        head_size = 128
        num_heads = 4
        num_filter = 4
        dropout = 0.1
                
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
        
    def call(self, x):
        
        inputs = x
        x = self.mha(x, x)  # attention layer
        self.dropout(x)
        x = x+inputs  # self.layer_norm(x+inputs)   No need for layer_norm if the number of feature column is 1.  
        res = x

        x = self.conv1D(x)    # Note conv1D has dimension NxTxD. conv2D has dimension NxT1xT2xD. 
        x = self.dropout(x)
        x = self.conv1D_2(x)  # This is needed to correct the dimension.
        x = x+res     # self.layer_norm(x+res)  
        
        return x
        
        
class transformer_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.num_transformer = 2
        dense_units = 64
        dropout = 0.1
        
        # x has dimension NxTxD.
        # GlobalAveragePooling1D will take the average of T dimension, and reduce array dimension to NxD.
        # "channels_first" is to reduce dimension to NxT.  "chaanels_last" is to reduce dimension to NxD.
        # We need to reduce the dimension because dense layer only accepts tensor of rank 2.
        # If you have multiple columns D, then you need to flatten T & D to make tensor rank 2.  NxTxD -> Nx(T*D) dimension.  
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")
        
        self.dense_layer = tf.keras.layers.Dense(dense_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.transformer_layer = transformer()
        self.output_layer = tf.keras.layers.Dense(D) 
        self.dense_layer2 = tf.keras.layers.Dense(dense_units)
        self.pos_encoding = self.positional_encoding( T, D )

    def positional_encoding(self, length, depth):
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

    def call(self, inputs, training):
        
        x = inputs
        x = tf.keras.layers.Add()([x, self.pos_encoding]) 
        
        # Note that after the transformer_layer, the input dimension stays the same.  
        for _ in range(self.num_transformer):  # stack number of transformer layers.
            x = self.transformer_layer(x)
        
        # If the input data is 2D, no need for self.pooling. 
        x = self.dense_layer2(x[:, -1, :])  # [:, -1, :] is needed to reduce dimension from NxTxD to NxD. -1 means output produce t+1 prediction.
        # x = self.pooling(x)  # Use this to reduce NxTxD to NxT if D=1.  
        x = self.dropout(x)
        outputs = self.output_layer(x)
        # note that -1 index is chosen to only predict t+1.
        # Another way is to flatten T & D to make tensor rank 2.  NxTxD -> Nx(T*D) dimension. 
        
        return outputs


#--------- Train the model ----------------------------------------------

model = transformer_model()  

model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam( learning_rate=0.001 ),
            #metrics=["sparse_categorical_accuracy"],
            )

rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  #"val_loss", 
                                            patience=20, verbose=1)
es = tf.keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True)



# test with input data.
dummy_x = np.random.rand( 1, T, D ) 
model( dummy_x ) 

res = model.fit(
                x_train,
                y_train,
                validation_split=0.2,
                epochs=1000,
                batch_size=64,
                callbacks=[ rlr, es ] ,
                )
        


model.predict( x_test[0:2] )
model.evaluate(x_test, y_test, verbose=1)

plt.plot( res.history['loss'], label='loss' ) 
plt.plot( res.history['val_loss'], label='val_loss' ) 
plt.legend()
plt.xlabel('epoch')
plt.show()


import pdb; pdb.set_trace()  




    

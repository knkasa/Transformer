# Example of timeseries Transformer with 1 feature.
# https://keras.io/examples/timeseries/timeseries_transformer_classification/ 

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

T = 10
D = 1

os.chdir('C:\my_working_env\deeplearning_practice')

#-------- checking ----------------
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


# Setup neural network with LSTM.
input_net = tf.keras.layers.Input(shape=(T,D,))   
hidden_net = tf.keras.layers.LSTM(50)(input_net)     
output_net = tf.keras.layers.Dense(D)(hidden_net)    
model = tf.keras.Model(inputs=input_net, outputs=output_net)

#my_loss = tf.keras.losses.MeanSquaredError()
#my_optimizer = tf.keras.optimizers.Adam(  )

model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam( learning_rate=0.001 ),
            #metrics=["sparse_categorical_accuracy"],
            )

rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss",  #"val_loss", 
                                            patience=30, verbose=0)
es = tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)

res = model.fit(
        encoder_train,
        y_train,
        validation_split=0.1,
        epochs=5000,
        batch_size=20,
        callbacks=[rlr, es],
        )

print()
for n in range(len(y_train)):
    print("prediction = ",  model.predict( encoder_train[n][tf.newaxis,:,:], verbose=False)[0,0], 
            " Target = ",  y_train[n].numpy() )
print("LSTM MSE*1000(train) = ", model.evaluate( encoder_train, y_train )*1000 )  

print()
for n in range(len(y_test)):
    print("prediction = ",  model.predict( encoder_test[n][tf.newaxis,:,:], verbose=False)[0,0], 
            " Target = ",  y_test[n].numpy() )
print("LSTM MSE*1000(test) = ", model.evaluate( encoder_test, y_test )*1000 )  



plt.plot( res.history['loss'], label='loss' ) 
plt.plot( res.history['val_loss'], label='val_loss' ) 
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('LSTM')
plt.show()

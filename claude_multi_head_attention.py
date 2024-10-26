# Multi-head layer attention example from Claude.
# Timeseries predictions with input features.

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import pandas as pd

# Positional Encoding
def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    
    angle_rates = 1 / (10000**depths)                # (1, depth)
    angle_rads = positions * angle_rates             # (seq, depth)
    
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# Multi-Head Attention Layer
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.query_dense = layers.Dense(units=d_model)
        self.key_dense = layers.Dense(units=d_model)
        self.value_dense = layers.Dense(units=d_model)
        
        self.dense = layers.Dense(units=d_model)
        
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]
        
        # Linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        
        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        scaled_attention_logits = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)
        
        return output

# Encoder Layer
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, name="encoder_layer"):
        super().__init__(name=name)
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training):
        attn_output = self.mha({
            'query': inputs,
            'key': inputs,
            'value': inputs
        })
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # you don't need layerNorm if input feature is only one.
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # you don't need layerNorm if input feature is only one.

# Generate sample data
def generate_sample_data(n_samples=1000):
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Generate multiple features
    temperature = np.sin(np.arange(n_samples) * 2 * np.pi / 365) * 20 + 15 + np.random.normal(0, 2, n_samples)
    humidity = np.cos(np.arange(n_samples) * 2 * np.pi / 365) * 30 + 60 + np.random.normal(0, 5, n_samples)
    wind_speed = np.abs(np.random.normal(10, 5, n_samples))
    pressure = np.random.normal(1013, 5, n_samples)
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure
    })
    return df

# Create and compile the model
def create_transformer_model(
    input_shape,
    d_model=64,
    num_heads=4,
    dff=256,
    num_encoder_layers=2,
    dropout_rate=0.1
):
    inputs = layers.Input(shape=input_shape)
    
    # Add positional encoding
    pos_encoding = positional_encoding(input_shape[0], input_shape[1])
    x = inputs + pos_encoding
    
    # Encoder layers
    for _ in range(num_encoder_layers):
        x = EncoderLayer(d_model, num_heads, dff, dropout_rate)(x, training=True)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final dense layers
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
def main():
    # Generate sample data
    df = generate_sample_data()
    
    # Prepare sequences
    sequence_length = 30
    features = ['temperature', 'humidity', 'wind_speed', 'pressure']
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(df) - sequence_length):
        sequences.append(df[features].iloc[i:i+sequence_length].values)
        targets.append(df['temperature'].iloc[i+sequence_length])
    
    X = np.array(sequences)
    y = np.array(targets)
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and compile model
    model = create_transformer_model(
        input_shape=(sequence_length, len(features)),
        d_model=64,
        num_heads=4,
        dff=256,
        num_encoder_layers=2
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"\nTest MAE: {test_mae:.2f}")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
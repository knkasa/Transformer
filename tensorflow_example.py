import tensorflow as tf
import numpy as np

# Sample time series data
num_samples = 1000  # Total number of samples
sequence_length = 20  # Length of each sequence
num_features = 5  # Number of features per time step

# Random data for input (features over time)
X = np.random.rand(num_samples, sequence_length, num_features).astype(np.float32)
# Random target for next time step prediction
y = np.random.rand(num_samples, num_features).astype(np.float32)

# Define the Transformer model with Encoder only
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # you don't need layerNorm if input feature is only one.
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define a model function
def create_encoder_only_transformer(sequence_length, num_features, embed_dim=32, num_heads=2, ff_dim=64, num_blocks=2):
    inputs = tf.keras.Input(shape=(sequence_length, num_features))
    x = tf.keras.layers.Dense(embed_dim)(inputs)

    # Stack multiple Transformer encoder blocks
    for _ in range(num_blocks):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_features)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model = create_encoder_only_transformer(sequence_length, num_features)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

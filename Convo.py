import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Activation,
    Flatten,
    Dense
)

model = Sequential([

    # Convolution Block 1
    Conv2D(32, (3,3), padding='same',
           input_shape=(64,64,3)),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(pool_size=(2,2)),

    # Convolution Block 2
    Conv2D(64, (3,3), padding='same'),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(pool_size=(2,2)),

    # Convolution Block 3
    Conv2D(128, (3,3), padding='same'),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(pool_size=(2,2)),

    # Fully Connected Layers
    Flatten(),

    Dense(256, activation='relu'),

    Dense(10, activation='softmax')
])

model.summary()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,)),
    tf.keras.layers.Dense(800, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dense(1200),
])
model.summary() 

class ConvoLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layer = nn.Linear(features, features)

    def forward(self, inputs):
        return self.layer(inputs)
class PoolingLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        return self.layer(inputs)    
    
inputs = nn.Linear(233, 456, bias=True)
sample_input = torch.randn(1, 233)
linear_output = inputs(sample_input)
convo_layer = ConvoLayer(features=linear_output.shape[-1])
convo_output = convo_layer(linear_output)

pooling_layer = PoolingLayer(
    in_features=convo_output.shape[-1],
    out_features=convo_output.shape[-1] // 2,
)
pooled_output = pooling_layer(convo_output)

print("Pooled output shape:", pooled_output.shape)
print(linear_output.shape)
print(convo_output.shape)    
    
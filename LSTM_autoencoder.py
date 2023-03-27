import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model

# Define the number of records and features
num_records = 41
num_features = 6

# Define the minimum and maximum length of each record
min_length = 100
max_length = 500

# Define mask value
mask_value = -1.0

# customized bottleneck to pass in mask (repeatvector doesn't handle mask itself)
class lstm_bottleneck(tf.keras.layers.Layer):
    def __init__(self, lstm_units, time_steps, **kwargs):
        self.lstm_units = lstm_units
        self.time_steps = time_steps
        self.lstm_layer = LSTM(lstm_units, return_sequences=False)
        self.repeat_layer = RepeatVector(time_steps)
        super(lstm_bottleneck, self).__init__(**kwargs)
    
    def call(self, inputs):
        # just call the two initialized layers
        return self.repeat_layer(self.lstm_layer(inputs))
    
    def compute_mask(self, inputs, mask=None):
        # return the input_mask directly
        return mask


# Generate random data with variable length
data = []
for _ in range(num_records):
    length = np.random.randint(min_length, max_length+1)
    record = np.random.rand(length, num_features) * 2 - 1
    data.append(record)

# Convert the data to a numpy array
data = np.array(data)

padded_train = tf.keras.preprocessing.sequence.pad_sequences(data, padding = 'post', value=mask_value)
samples, timesteps, features = padded_train.shape

# Print the shape of the data
print("Padded data shape:", padded_train.shape)

# Define the input shape
input_shape = (timesteps, features)


## model 1 - loss = zero for random data, works on real data
# experiment = 1
# input_seq = Input(shape = (timesteps, features))
# masking_layer = Masking(mask_value = mask_value)(input_seq)
# lstm_layer = LSTM(32, return_sequences=True)(masking_layer)
# output_layer = TimeDistributed(Dense(features))(lstm_layer)
# auto_encoder = Model(inputs = input_seq, outputs = output_layer)

## model 2: LSTM model: loss = zero for random data, works on real data
# experiment = 2
# auto_encoder = tf.keras.models.Sequential()
# auto_encoder.add(Masking(mask_value=mask_value, input_shape=(timesteps, features)))
# auto_encoder.add(LSTM(20, activation='tanh',return_sequences=True))
# auto_encoder.add(LSTM(15, activation='tanh', return_sequences=True))
# auto_encoder.add(LSTM(5, activation='tanh', return_sequences=True))
# auto_encoder.add(LSTM(15, activation='tanh', return_sequences=True))
# auto_encoder.add(LSTM(20, activation='tanh', return_sequences=True))
# auto_encoder.add(TimeDistributed(Dense(features,activation='tanh')))

## model 3: LSTM autoencoder: loss = zero for random data, works on real data
# experiment = 3
# auto_encoder = tf.keras.models.Sequential()
# auto_encoder.add(Masking(mask_value=mask_value, input_shape=(timesteps, features)))
# auto_encoder.add(LSTM(20, activation='tanh',return_sequences=True))
# auto_encoder.add(LSTM(15, activation='tanh', return_sequences=False))
# auto_encoder.add(RepeatVector(timesteps)) 
# auto_encoder.add(LSTM(15, activation='tanh', return_sequences=True))
# auto_encoder.add(LSTM(20, activation='tanh', return_sequences=True))
# auto_encoder.add(TimeDistributed(Dense(features,activation='tanh')))

## model 4: use custom bottle neck for no need to use sample weight in model.fit
experiment = 4
auto_encoder = tf.keras.models.Sequential()
auto_encoder.add(Masking(mask_value=mask_value, input_shape=(timesteps, features)))
auto_encoder.add(LSTM(20, activation='tanh',return_sequences=True))
auto_encoder.add(LSTM(15, activation='tanh', return_sequences=True))
auto_encoder.add(lstm_bottleneck(15, timesteps)) 
auto_encoder.add(LSTM(15, activation='tanh', return_sequences=True))
auto_encoder.add(LSTM(20, activation='tanh', return_sequences=True))
auto_encoder.add(TimeDistributed(Dense(features,activation='tanh')))


print(auto_encoder.summary())

# validate the masking procedure
for i, l in enumerate(auto_encoder.layers):
    print(f'layer {i}: {l}')
    print(f'has input mask: {l.input_mask}')
    print(f'has output mask: {l.output_mask}')


## compile
auto_encoder.compile(optimizer='adam', loss ='mse',sample_weight_mode='temporal')
sample_weight = np.not_equal(padded_train[:,:,0],mask_value).astype(float)
## fit
if experiment in {1,2,3}:
    auto_encoder.fit(padded_train, padded_train, epochs= 10, batch_size = 10, validation_split = 0.2, sample_weight=sample_weight)
elif experiment == 4:
    auto_encoder.fit(padded_train, padded_train, epochs= 10, batch_size = 10, validation_split = 0.2)

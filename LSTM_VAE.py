import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, RepeatVector, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Define the number of records and features
num_records = 41
num_features = 6

# Define the minimum and maximum length of each record
min_length = 100
max_length = 500

apply_masking = True 


# Generate random data with variable length
data = []
mask_value = -1.0
for _ in range(num_records):
    length = np.random.randint(min_length, max_length+1)
    record = np.random.rand(length, num_features) * 2 - 1
    data.append(record)

# Convert the data to a numpy array
data = np.array(data)
x_train = data

padded_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding = 'post', value=mask_value, dtype='float64')
samples, timesteps, features = padded_train.shape


# Define the input shape
input_shape = (timesteps, features)

# Define the number of latent dimensions
latent_dim = 10
inter_dim = 32


# Define the sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch_size = tf.shape(z_mean)[0]
    epsilon = K.random_normal(shape=(batch_size,latent_dim), mean=0., stddev=1.)
    return z_mean + z_log_var * epsilon # or  K.exp(0.5 * z_log_var)

# Define the encoder model
inputs = Input(shape=input_shape, name= 'input_layer')

# intermediate dimension
if apply_masking:
    h = Masking(mask_value=mask_value, name = 'input_masking')(inputs)
    h = LSTM(2*inter_dim, activation='tanh', return_sequences= True, name ='LSTM_intermediate', kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(h)
    h = LSTM(inter_dim, activation='tanh')(h)
else:
    h = LSTM(inter_dim, activation='tanh', name ='LSTM_intermediate',  kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(inputs)

# z layer
z_mean = Dense(latent_dim, name = 'z_mean')(h)
z_log_var = Dense(latent_dim, name = 'z_variance')(h)
z = Lambda(sampling, name='sampling')([z_mean, z_log_var])

# Define the decoder model

decoder = RepeatVector(timesteps, name = 'decoder_repeatvector')(z) # method 1, required sample weight in the fit method
decoder = LSTM(inter_dim, activation='tanh', return_sequences=True, name='decoder_LSTM', kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(decoder)
decoder = LSTM(features, activation='tanh', return_sequences=True, name='decoder_LSTM_f', kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01))(decoder) # new
if apply_masking:
    decoder = Masking(mask_value=mask_value, name = 'decoder_masking')(decoder)
decoder = TimeDistributed(Dense(features), name='decoder_timedist')(decoder)

# Define the VAE loss function
def vae_loss(inputs, outputs, z_args):
    z_mean, z_log_var = z_args
    # E(log P(x|z))

    if apply_masking:
        # Assuming the mask is named "mask"
        mask = Masking(mask_value=mask_value)(inputs)
        # Calculate reconstruction loss
        squared_diff = K.square(inputs - outputs)
        unmasked_squared_diff = squared_diff * K.cast(K.not_equal(mask, mask_value), K.floatx())
        # dividing purpose: avoid bias towards longer or shorter sequences, since longer sequences tend to have more masked elements and therefore a smaller reconstruction loss if unnormalized
        reconstruction_loss = K.sum(unmasked_squared_diff) / K.sum(K.cast(K.not_equal(mask, mask_value), K.floatx()))
    else:
        reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)) #check without axis  axis=-1
    return K.mean(reconstruction_loss + kl_loss)




# Define the VAE model
vae = Model(inputs, decoder)
print(vae.summary())

# Add custom loss
vae.add_loss(vae_loss(inputs, decoder, (z_mean, z_log_var)))

# Compile
vae.compile(optimizer='adam')


# Validate the masking procedure
for i, l in enumerate(vae.layers):
    print(f'--------------layer {i}: {l.name}')
    print(f'has input mask: {l.input_mask}')
    print(f'has output mask: {l.output_mask}')


# Train the VAE model
if apply_masking: 
    sample_weight = np.not_equal(padded_train[:,:,0],mask_value).astype(float)
    vae.fit(padded_train[10:], epochs=800, batch_size=64, validation_data= (padded_train[:10],None), sample_weight=sample_weight)
else:    
    vae.fit(padded_train[10:], epochs=800, batch_size=64, validation_data= (padded_train[:10], None))

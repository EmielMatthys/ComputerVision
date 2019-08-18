import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
import cv2
import numpy as np
import random
from keras.callbacks import TensorBoard

#Conventie: genereer x_train, x_val 1 keer en houdt in spyder variable storage
#          gebruik X_train en X_store voor aanpassingen.
if not 'x_train' in globals():
    from project_dataset_script_v2 import x_train, x_val, filter
    
print(x_train.shape)    
num_classes = 5
X_train = x_train
X_val = x_val
#X_train_gray_flat = np.array(X_train_gray)

n_epochs = 1
n_batches = 256

n, img_rows, img_cols, n_channels = x_train.shape

#inputs = Input(shape=(img_rows, img_cols, n_channels))
#layer = Conv2D(64, (3, 3), padding='same')(inputs)
#layer = BatchNormalization()(layer)
#layer = Activation('relu')(layer)
#layer = MaxPooling2D((2, 2), padding='same')(layer)
#layer = Conv2D(32, (3, 3), padding='same')(layer)
#layer = BatchNormalization()(layer)
#layer = Activation('relu')(layer)
#layer = MaxPooling2D((2, 2), padding='same')(layer)
#layer = Conv2D(16, (3, 3), padding='same')(layer)
#layer = BatchNormalization()(layer)
#layer = Activation('relu')(layer)
#encoded = MaxPooling2D((2, 2), padding='same', name='code')(layer)
#
#layer = Conv2D(16, (3, 3), padding='same')(encoded)
#layer = BatchNormalization()(layer)
#layer = Activation('relu')(layer)
#layer = UpSampling2D((2, 2))(layer)
#layer = Conv2D(32, (3, 3), padding='same')(layer)
#layer = BatchNormalization()(layer)
#layer = Activation('relu')(layer)
#layer = UpSampling2D((2, 2))(layer)
#layer = Conv2D(64, (3, 3), padding='same')(layer)
#layer = BatchNormalization()(layer)
#layer = Activation('relu')(layer)
#layer = UpSampling2D((2, 2))(layer)
#layer = Conv2D(3, (3, 3), padding='same')(layer)
#layer = BatchNormalization()(layer)
#decoded = Activation('sigmoid')(layer)
#
#conv_autoencoder = Model(inputs, decoded)
#conv_autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['mean_squared_error'])
#
#history = conv_autoencoder.fit(x_train, x_train,epochs=n_epochs,batch_size=n_batches,verbose=1,
#                               validation_data=(x_val, x_val),shuffle=True)


input_img = Input(shape=(img_rows, img_cols, n_channels))

x = Conv2D(64, 3, activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(32, 3, activation='relu', padding='same')(x)
#x = MaxPooling2D((2,2), padding='same')(x)
#x = Conv2D(16, 3, activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

#x = Conv2D(16, 3, activation='relu', padding='same')(encoded)
#x = UpSampling2D((2,2))(x)
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
#x = UpSampling2D((2,2))(x)
decoded = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

#Model initialization and compile
print('Now compiling autoencoder')
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, 
                epochs=n_epochs,
                batch_size=n_batches,
                verbose=1,
                validation_data=(x_val,x_val),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                )

#model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['mean_squared_error'])
#
#history = model.fit(x_train, x_train,epochs=n_epochs,batch_size=n_batches,verbose=1,
#                             validation_data=(x_val, x_val),shuffle=True)
#exit(0)
#random_indices = [random.randint(0, x_val.shape[0]) for i in range(10)] 
#reconstruction_error_list = []
#val_images = conv_autoencoder.predict(x_val)
#fig = plt.figure(figsize=(25,5))
#ax = fig.subplots(2, 10)
#i=0
#for index in random_indices:
#    dec = val_images[index]
#    act = x_val[index].astype('float32')/255.0
#    reconstruction_error_list.append(np.mean(((dec-act)*(dec-act))))
#    ax[0,i].imshow(act, cmap='gray')
#    ax[0,i].axis('off')
#    ax[0,i].set_title('Original Image', fontdict={'fontsize': 12, 'fontweight': 'medium'})
#    ax[1,i].imshow(dec, cmap='gray')
#    ax[1,i].axis('off')
#    ax[1,i].set_title('Reconstructed Image', fontdict={'fontsize': 12, 'fontweight': 'medium'})
#    i = i+1
#print('Reconstruction error on entire dataset: ' + str(np.round(np.sqrt(np.mean(reconstruction_error_list)),4)))
#plt.show()
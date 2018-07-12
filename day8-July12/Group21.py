
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
import scipy
import skimage
from skimage.transform import resize
import cv2


# In[2]:


print(os.listdir("/Users/kasspk/Documents/AUAISummer2018/chest_xray/"))


# In[3]:


data = "/Users/kasspk/Documents/AUAISummer2018/chest_xray/"
train_data = "/Users/kasspk/Documents/AUAISummer2018/chest_xray/train/"
test_data = "/Users/kasspk/Documents/AUAISummer2018/chest_xray/test/"
val_data = "/Users/kasspk/Documents/AUAISummer2018/chest_xray/val/"


# In[4]:


print(os.listdir(test_data))


# In[5]:


def get_data(Dir):
    X = []
    y = []
    
    for next_dir in os.listdir(Dir):
        if not next_dir.startswith("."):
            if next_dir in ['NORMAL']:
                label = 0
            elif next_dir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
        
            temp = Dir + next_dir
        
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp+'/'+file)
                if img is not None:
                    img = skimage.transform.resize(img,(100,100,3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)
        
    return X,y


# In[6]:


X_train, y_train = get_data(train_data)


# In[7]:


X_test, y_test = get_data(test_data)


# In[8]:


X_train = np.asarray(X_train)


# In[9]:


print(X_train.shape)


# In[10]:


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


# In[11]:


print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# In[12]:


print(y_train[5000])


# In[13]:


from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


# In[14]:


print(y_train[5000])


# In[15]:


print(y_train[0])


# In[16]:


from tempfile import TemporaryFile
outfile = TemporaryFile()
np.savez(outfile, X_train, y_train, X_test, y_test)


# In[17]:


outfile.seek(0)
npzfile = np.load(outfile)
npzfile.files
npzfile['arr_0']


# In[18]:


Pimages = os.listdir(train_data+'PNEUMONIA')
Nimages = os.listdir(train_data+'NORMAL')
Pimages[1]


# In[19]:


from matplotlib import pyplot as plt

img = cv2.imread(train_data+'NORMAL/'+ Nimages[1],0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


# In[20]:


img = cv2.imread(train_data+'PNEUMONIA/'+ Pimages[1],0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


# In[21]:


X_train1 = npzfile['arr_0']


# In[31]:


import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint

filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])


# In[32]:


from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD , RMSprop
from keras.layers import Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('tf')

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(100,100,3)))
model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(100,100,3)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2 , activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.00005),
                  metrics=['accuracy'])

print(model.summary())


# In[33]:


history = model.fit(X_train, y_train, validation_data = (X_test , y_test) ,callbacks=[checkpoint] ,
          epochs=5,verbose=2)


# In[34]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[40]:


from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)


# In[41]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_true, pred)


# In[42]:


acc = (134+379)/(134+100+11+379)
print(acc)


# Classificação de imagens que pode determinar se uma imagem representa 
# um gesto de sinal positivo ou negativo (:thumbsup: ou :thumbsdown:).
# Autora: Franciele Cicconet
# Data: Julho/2020

#import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input, Model
from skimage.io import imread, imsave
import os, shutil

from skimage import data, color
from skimage.transform import rescale, resize


print('-------------------------')
print('Parâmetros')


batchSize = 8
learningRate = 0.0001
nIterations = 100

imSize = 64
nChannels = 1   
nClasses = 2
nImagesPerClass = [30,30] 

print('-------------------------')
print('Modelo')


x = Input(batch_shape=(None,imSize,imSize,nChannels))

conv = Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=(imSize,imSize,nChannels))(x) # 16
maxp = MaxPooling2D(pool_size=(2, 2))(conv)

flat = Flatten()(maxp)
fc = Dense(4096, activation='relu')(flat) 
sm = Dense(nClasses, activation='softmax')(fc)
model = Model(x,sm)


model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.train.AdamOptimizer(learning_rate = learningRate),
              metrics=['accuracy'])

print('-------------------------')
print('Treinando')

# Processamento das imagens
def preProcess(image):
    image = color.rgb2gray(image)
    image = rescale(image, imSize/128) 
    image = np.reshape(image,(image.shape[0], image.shape[1], 1))
    image = image-np.mean(image)
    image = image/np.std(image)
    return image

def getBatch():
    x_batch = np.zeros((batchSize,imSize,imSize,nChannels))
    y_batch = np.zeros((batchSize,nClasses))
    for j in range(batchSize):
        cIndex = np.random.randint(nClasses)
        iIndex = np.random.randint(nImagesPerClass[cIndex])
 #       print('im_data/train/%d/im%d.jpeg' %(cIndex,iIndex))
        image = imread('im_data/train/%d/im%d.jpeg' %(cIndex,iIndex)).astype('double')/255
        image = preProcess(image)
        if np.random.rand() < 0.5: # aumentando o conjunto de treinamento ao espelhar
            image = np.fliplr(image)
        x_batch[j,:,:,:] = image
        y_batch[j,cIndex] = 1 
    return x_batch, y_batch 

#getBatch()

for i in range(nIterations):
    x_batch, y_batch = getBatch()
    
    # import sys; sys.exit(0)
    model.train_on_batch(x_batch, y_batch)

    loss_and_metrics = model.evaluate(x_batch, y_batch, batch_size=batchSize, verbose=0)
    print('step: %d, loss: %f, acc: %f' % (i,loss_and_metrics[0],loss_and_metrics[1]))

print('-------------------------')
print('Testando')

def getTest():
    x_batch = np.zeros((20,imSize,imSize,nChannels))
    y_batch = np.zeros((20,nClasses))
    for j in range(10):
        image = imread('im_data/test/0/im%d.jpeg' % j).astype('double')/255
        image = preProcess(image)
        x_batch[j,:,:,:] = image
        y_batch[j,0] = 1
        image = imread('im_data/test/1/im%d.jpeg' % j).astype('double')/255
        image = preProcess(image)
        x_batch[10+j,:,:,:] = image
        y_batch[10+j,1] = 1
    return x_batch, y_batch

#getTest()

x_test, y_test = getTest()

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
print('test acc: %f' % loss_and_metrics[1])


print('Imagem | Classificação')
prediction = model.predict(x_test)
for i in range(prediction.shape[0]):
#    print(i,'|', 'thumbsdown probability:', '%.2f,' % prediction[i,0], 'thumbsup probability:', '%.2f' % prediction[i,1])
    if prediction[i,0] > 0.5:
        print(i, '| thumbsdown')
    else:
        print(i,'| thumbsup')
        
        
        
        
        
        

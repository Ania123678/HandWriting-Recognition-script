
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils, optimizers
import cv2

# data set for digit writing
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalization
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#basic neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28,1))) #zamienia "tablice" w jeden "pasek" danych
model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation = tf.nn.sigmoid)) #output

model.compile(optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)
loss, accuracy,  = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')
for x in range(1,4):
    img = cv2.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'to jest prawdopodobnie:', np.argmax(prediction))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()





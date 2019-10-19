import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist #28*28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1) #normalization
x_test = tf.keras.utils.normalize(x_test, axis=1) #normalization

model = tf.keras.models.Sequential() #sequential models, it's a feed-forward like the image we drew
model.add(tf.keras.layers.Flatten()) #using flatten as input layer just to make our lives easier
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #how many unit in the layer, we're gonna use 128 neurons in the layer. activation function is what is gonna make that neuron fire or sort of fire. relu is rectified linear
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #output layer, if it's in the case of classification then it'll have your number of class. activation function using softmax for a probability distribution

#parameter for model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3) #training model

#neural netword doesn't actually attempt to optimize for accuracy, it's doesn't try to maximize accuracy, it's always trying to minimize loss
#so the way you calculate loss can make a huge impact because it's what's losses relationship to your accuracy optimizer
#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()

#the goal: the model is actually generalizing, rather memorizing it

#calculate validation loss and validation accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('epic_num_reader.model') #save the model
new_model = tf.keras.models.load_model('epic_num_reader.model') #load existing model

predictions = new_model.predict([x_test])
print(predictions)

print(np.argmax(predictions[0])) #print prediction at index 0

plt.imshow(x_test[0])
plt.show()

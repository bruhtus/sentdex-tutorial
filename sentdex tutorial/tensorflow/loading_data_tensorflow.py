import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle #to save your shuffle dataset

source = 'PetImages'
categories = ['Dog', 'Cat']

for category in categories: #show grayscale image
    path = os.path.join(source, category) #path to cats or dogs directory
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
       #plt.imshow(img_array, cmap='gray')
       #plt.show()
        break
    break

print(img_array.shape)

img_size = 50

new_array = cv2.resize(img_array, (img_size, img_size))
print(new_array.shape)

#plt.imshow(new_array, cmap='gray')
#plt.show()

training_data = []

def create_training_data():
    for category in categories: #show grayscale image
        path = os.path.join(source, category) #path to cats or dogs directory
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data)) #print length training_data

random.shuffle(training_data) #shuffling training_data

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1) #-1 just to pass whatever value in it, and 1 because of grayscale

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

print(X[1])

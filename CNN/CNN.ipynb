from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras import datasets, layers, models,metrics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from packaging import version
%load_ext tensorboard
# !rm -rf ./logs/ 
from keras.preprocessing.image import ImageDataGenerator
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
)
datagen.fit(train_images)

history = model.fit(datagen.flow(train_images, train_labels), epochs=10, verbose=0,
                    validation_data=(test_images, test_labels),
                    callbacks=[tensorboard_callback])

# history = model.fit(train_images, train_labels, epochs=10, verbose=0,
#                     validation_data=(test_images, test_labels),
#                     callbacks=[tensorboard_callback])

print(history.history)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# !rm -rf ./logs/

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2 )

y_pred=model.predict_classes(test_images)
print(y_pred)
# tf.compat.v1.metrics.mean_per_class_accuracy(test_labels,
#     y_pred,
#     [0,1,2,3,4,5,6,7,8,9])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
con_mat = confusion_matrix(test_labels, y_pred)
print(con_mat)
plt.figure()
plt.imshow(con_mat, interpolation='nearest')
print()
# avg Accuracy of Each Class
print("Average accuracy of each class")
print()
for i in range(0,10):
  print("Class ",class_names[i]," : ", con_mat[i][i]/10," %")

  # more matrix imformation

print()
print(classification_report(test_labels, y_pred, target_names=class_names))


print("Validation Accuracy: ",test_acc)
%tensorboard --logdir logs/scalars
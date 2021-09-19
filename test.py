import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

PATH = "data"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 50
IMG_SIZE = (100, 100)

# Load the images
train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
validation_dataset = image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

class_names = train_dataset.class_names
print("Classes: ", class_names)
train_batches = tf.data.experimental.cardinality(validation_dataset)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# Take a small portion of the training dataset for validation
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 2)
validation_dataset = validation_dataset.skip(val_batches // 2)

print('Number of Train batches: %d' % tf.data.experimental.cardinality(train_dataset))
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
#input("")

# Buffer dataset to speed up trainingdf
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

model = tf.keras.models.load_model("modelo_salvo")

list1 = list()
list2 = list()
mLoss = 0
mAcc = 0
for i in range(5):
    # Evaluate model
    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    # input("")

    # Test model in a batch of images
    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Predictions:\n', predictions.numpy())
    print('Labels:\n', label_batch)

    list1 = np.append(list1, predictions.numpy())
    list2 = np.append(list2, label_batch)
    mLoss = mLoss + loss
    mAcc = mAcc + accuracy

mLoss = mLoss / 5
mAcc = mAcc / 5

print("Loss media = ", mLoss)
print("Acc media = ", mAcc)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
plt.show()


#print('Predictions:\n', predictions.numpy())
for i in predictions.numpy():
    print(class_names[i])
print("")

print('Labels:\n', )
for i in label_batch:
    print(class_names[i])
print("")

y_actu = pd.Series(list2, name='Real')
y_pred = pd.Series(list1, name='Previsto')
df_confusion = pd.crosstab(y_actu, y_pred)
#df_conf_norm = df_confusion / df_confusion.sum(axis=1)

sns.heatmap(df_confusion, cmap='Blues', annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.show()

#sns.heatmap(df_conf_norm, cmap='Blues', annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
#plt.show()

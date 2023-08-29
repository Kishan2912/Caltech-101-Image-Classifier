# %%
import tensorflow as tf 
import pickle
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Input,Dense,Flatten
from keras.optimizers import Adam


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
import os

os.makedirs("/kaggle/working/results/", exist_ok=True)
tf.random.set_seed(0)

img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/Group_16/train",
  validation_split=0,
  image_size=(img_height, img_width),
  label_mode="int",
  batch_size=250)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/Group_16/val",
  validation_split=0,
  image_size=(img_height, img_width),
  label_mode="int",
  batch_size=50)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "/Group_16/test",
  image_size=(img_height, img_width),
  label_mode="int",
  batch_size=100)


train_ds = train_ds.cache()
val_ds = val_ds.cache()
test_ds = test_ds.cache()

# %%
import tensorflow as tf

input_shape = (224, 224, 3)
num_classes = 5

model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (11,11), strides=4, padding='valid', activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(16, (5,5), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# %%
model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (11,11), strides=4, padding='valid', activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(16, (5,5), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(32, (3,3), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# %%
model3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (11,11), strides=4, padding='valid', activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(16, (5,5), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(32, (3,3), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# %%
trainX, trainY = next(iter(train_ds))
valX, valY = next(iter(val_ds))
testX, testY = next(iter(test_ds))

# %%
# Compile the model1
model1.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Train the model1
history=model1.fit(train_ds, epochs=100,
                                callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(
                                        f"/kaggle/working/models/model1.hdf5",
                                        verbose=0
                                    )
                                ],shuffle=True,verbose=1, validation_data=val_ds)

results = model1.evaluate(val_ds)

with open(f"/kaggle/working/results/model1.pkl", "wb") as file:
        pickle.dump((history), file)

# %%
# Compile the model1
model2.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Train the model1
history=model2.fit(train_ds, epochs=100,
                                callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(
                                        f"/kaggle/working/models/model2.hdf5",
                                        verbose=0
                                    )
                                ],shuffle=True,verbose=1, validation_data=val_ds)

results = model2.evaluate(val_ds)

with open(f"/kaggle/working/results/model2.pkl", "wb") as file:
        pickle.dump((history), file)

# %%
# Compile the model1
model3.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


# Train the model1
history=model3.fit(train_ds, epochs=100,
                                callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(
                                        f"/kaggle/working/models/model3.hdf5",
                                        verbose=0
                                    )
                                ],shuffle=True,verbose=1, validation_data=val_ds)

results = model3.evaluate(val_ds)

with open(f"/kaggle/working/results/model3.pkl", "wb") as file:
        pickle.dump((history), file)

# %%
image_tensor=trainX[0].numpy().astype("uint8")
plt.imshow(image_tensor)
plt.axis('off')
plt.show()

# %%
from sklearn.metrics import accuracy_score

predictions = model1.predict(val_ds)
predictions = tf.math.argmax(predictions, axis=1)
print(accuracy_score(valY, predictions))
cm = tf.math.confusion_matrix(valY, predictions, num_classes=5).numpy()
cm = cm.astype(str)
print("\n".join([f"&\\textbf{{{i+1}}} &" + " &".join(list(cm[i])) + "\\\\ \cline{3-7}" for i in range(5)]))
print(cm)

# %%
from sklearn.metrics import accuracy_score

predictions = model2.predict(val_ds)
predictions = tf.math.argmax(predictions, axis=1)
print(accuracy_score(valY, predictions))
cm = tf.math.confusion_matrix(valY, predictions, num_classes=5).numpy()
cm = cm.astype(str)
print("\n".join([f"&\\textbf{{{i+1}}} &" + " &".join(list(cm[i])) + "\\\\ \cline{3-7}" for i in range(5)]))
print(cm)

# %%
from sklearn.metrics import accuracy_score

predictions = model3.predict(val_ds)
predictions = tf.math.argmax(predictions, axis=1)
print(accuracy_score(valY, predictions))
cm = tf.math.confusion_matrix(valY, predictions, num_classes=5).numpy()
cm = cm.astype(str)
print("\n".join([f"&\\textbf{{{i+1}}} &" + " &".join(list(cm[i])) + "\\\\ \cline{3-7}" for i in range(5)]))
print(cm)

# %%
from sklearn.metrics import accuracy_score

predictions = model3.predict(test_ds)
predictions = tf.math.argmax(predictions, axis=1)
print(accuracy_score(testY, predictions))
cm = tf.math.confusion_matrix(testY, predictions, num_classes=5).numpy()
cm = cm.astype(str)
print("\n".join([f"&\\textbf{{{i+1}}} &" + " &".join(list(cm[i])) + "\\\\ \cline{3-7}" for i in range(5)]))
print(cm)

# %%
model1.summary()

# %%
model2.summary()

# %%
model3.summary()

# %%
new_model = tf.keras.models.Model(inputs=model3.inputs, outputs=model3.get_layer('conv2d_8').output)

features = new_model.predict(tf.expand_dims(image_tensor, axis=0))

num_filters = features.shape[-1]

print(features.shape)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < num_filters:
        m = features[0, :, :, i]
        f_min, f_max = m.min(), m.max()
        m = (m - f_min) / (fvalY_max - f_min + 1e-8)
        ax.imshow(m, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# %%
# Create a new model that includes only the layers up to and including the first convolutional layer
new_model1 = tf.keras.models.Sequential(model3.layers[:3])
new_model2 = tf.keras.models.Sequential(model3.layers[:5])

features1 = new_model1.predict(tf.expand_dims(image_tensor, axis=0))
features2 = new_model2.predict(tf.expand_dims(image_tensor, axis=0))

# Assuming `features` is the output of the code snippet in my previous response
num_filters1 = features1.shape[-1]
num_filters2 = features2.shape[-1]
# Create a new model that includes only the layers up to and including the first convolutional layer
new_model1 = tf.keras.models.Sequential(model3.layers[:3])
new_model2 = tf.keras.models.Sequential(model3.layers[:5])

features1 = new_model1.predict(tf.expand_dims(image_tensor, axis=0))
features2 = new_model2.predict(tf.expand_dims(image_tensor, axis=0))

# Assuming `features` is the output of the code snippet in my previous response
num_filters1 = features1.shape[-1]
num_filters2 = features2.shape[-1]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < 8:
        ax.imshow(features1[0, :, :, i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
plt.close()

print("\n\n")

# Plot the feature maps from 5th convulutional layer
fig, axes = plt.subplots(4, num_filters2//4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < num_filters2:
        ax.imshow(features2[0, :, :, i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
plt.close()
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < 8:
        ax.imshow(features1[0, :, :, i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
plt.close()

print("\n\n")

# Plot the feature maps from 5th convulutional layer
fig, axes = plt.subplots(4, num_filters2//4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < num_filters2:
        ax.imshow(features2[0, :, :, i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
plt.close()

# %%
five_images = tf.keras.preprocessing.image_dataset_from_directory(
  "/kaggle/input/cs671-ass5-five-images/5_images",
  validation_split=0,
  image_size=(img_height, img_width),
  label_mode="int",
  batch_size=25)

five_images = five_images.cache()
fiveX, fiveY = next(iter(five_images))


# %%
import tensorflow.keras.backend as K

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# %%
import cv2

for i in range(25):
    
    x = np.expand_dims(fiveX[i, ...].numpy(), axis=0)
    orig = x[0].astype(int)
    
    gb_model = Model(
        inputs = [model3.inputs],
        outputs = [model3.get_layer("conv2d_8").output]
    )
    
    new_layer = tf.keras.layers.Lambda(lambda x : tf.cast(tf.equal(x, tf.reduce_max(x)), tf.float32)*x)(gb_model.output)
    
    dummy_model = Model(
        inputs=gb_model.inputs,
        outputs=[new_layer]
    )
    
    with tf.GradientTape() as tape:
        inputs = tf.cast(x, tf.float32)
        tape.watch(inputs)
        outputs = dummy_model(inputs)

    grads = tape.gradient(outputs,inputs)
    sm = deprocess_image(np.array(grads))[0]
    
    gs = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY)
    th = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(th)
    x,y,w,h = cv2.boundingRect(coords)
    if (x==0 and y==0):
        th = ~th
        coords = cv2.findNonZero(th)
        x,y,w,h = cv2.boundingRect(coords)
    
    mi = np.argmax(gs)
    mi = np.unravel_index(mi, gs.shape)
    nw = cv2.rectangle(orig.astype(np.uint8), (x,y), (x+w,y+h), (0,255,0), 2)
    
    fig, ax = plt.subplots(1, 4)
    [ax_.axis('off') for ax_ in ax]
    ax[0].imshow(sm)
    ax[1].imshow(orig)
    ax[2].imshow(nw)
    ax[3].imshow(th)
    plt.show()
    plt.close()
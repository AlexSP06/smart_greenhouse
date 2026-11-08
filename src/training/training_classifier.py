import json
from pathlib import Path
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

TRAIN_DIR = "data/train_15"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
IMG_SIZE = (224, 224)
BATCH = 32 
EPOCHS = 15

train_ds = keras.utils.image_dataset_from_directory(TRAIN_DIR, validation_split=0.2, subset="training",
                                                    seed=42, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = keras.utils.image_dataset_from_directory(TRAIN_DIR, validation_split=0.2, subset="validation",
                                                  seed=42, image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
print("Classes: ", class_names)
(MODEL_DIR / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          callbacks=[keras.callbacks.EarlyStopping(
              patience=3, restore_best_weights=True
          )])

model.save(MODEL_DIR / "plant_classifier.keras")
print("Model saved.")
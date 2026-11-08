import json
from tensorflow import keras

model = keras.models.load_model("models/plant_classifier.keras")
class_names = json.load(open("models/class_names.json"))

test_ds = keras.utils.image_dataset_from_directory(
    "data/test", image_size=(224, 224), batch_size=32,
    class_names=class_names, shuffle=False
)

loss, acc = model.evaluate(test_ds)
print(f"Accuracy of the dataset: {acc:.2%}")
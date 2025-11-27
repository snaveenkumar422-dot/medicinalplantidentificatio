import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for generating images without display
import matplotlib.pyplot as plt

# Load and preprocess images
def load_images(image_dir, image_size=(299, 299)):
    images, labels = [], []
    class_mapping = {}  # Class mapping will be dynamic based on folder names

    for label_dir in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_dir)
        if os.path.isdir(label_path):
            class_mapping[label_dir] = len(class_mapping)  # Assign an integer to each class
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        image_path, target_size=image_size
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_mapping[label_dir])
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")

    print(f"Loaded {len(images)} images and {len(labels)} labels.")
    return np.array(images), np.array(labels), class_mapping

# Set paths
image_dir = r"E:\lance colin y\PROJECT\Annamalai final year project\Dataset\train\Batch_5"

# Load dataset
images, labels, class_mapping = load_images(image_dir)

# Print class names
print("Class Mapping (Class Names and their Numeric Labels):")
for class_name, label in class_mapping.items():
    print(f"{label}: {class_name}")

# Check if images and labels are consistent
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels loaded. Check the dataset path and structure.")

# Normalize images
images = images / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)
datagen.fit(X_train)

# Xception model (modified for custom classes)
def create_xception_model(input_shape=(299, 299, 3), num_classes=200):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# Compile and train Xception model
xception_model = create_xception_model(num_classes=len(class_mapping))
xception_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = xception_model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_test, y_test),
    epochs=50
)

# Save the model
xception_model.save("xception_medicinal_plants_batch_5.h5")
print("Model saved as xception_medicinal_plants_batch_5.h5")

# Evaluate the model
y_pred = np.argmax(xception_model.predict(X_test), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
n_classes = len(class_mapping)

y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
y_pred_proba = xception_model.predict(X_test)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ({list(class_mapping.keys())[i]}) (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve4.png')  # Save the ROC curve plot

# Plot loss and accuracy curves
plt.figure(figsize=(12, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('loss_accuracy4.png')  # Save the loss and accuracy plot

plt.show()  # Show the plots (optional, as the plots are already saved)
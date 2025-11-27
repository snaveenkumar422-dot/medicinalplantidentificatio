import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# Define class mapping
class_mapping = {
    0: "Abelmoschus sagittifolius", 1: "Abrus precatorius", 2: "Abutilon indicum",
    3: "Acanthus integrifolius", 4: "Acorus tatarinowii", 5: "Agave americana",
    6: "Ageratum conyzoides", 7: "Allium ramosum", 8: "Alocasia macrorrhizos",
    9: "Aloe vera", 10: "Alpinia officinarum", 11: "Amomum longiligulare",
    12: "Ampelopsis cantoniensis", 13: "Andrographis paniculata", 14: "Angelica dahurica",
    15: "Ardisia sylvestris", 16: "Artemisia vulgaris", 17: "Artocarpus altilis",
    18: "Artocarpus heterophyllus", 19: "Artocarpus lakoocha", 20: "Asparagus cochinchinensis",
    21: "Asparagus officinalis", 22: "Averrhoa carambola", 23: "Baccaurea sp",
    24: "Barleria lupulina", 25: "Bengal Arum", 26: "Berchemia lineata",
    27: "Bidens pilosa", 28: "Bischofia trifoliata", 29: "Blackberry Lily",
    30: "Blumea balsamifera", 31: "Boehmeria nivea", 32: "Breynia vitis",
    33: "Caesalpinia sappan", 34: "Callerya speciosa", 35: "Callisia fragrans",
    36: "Calophyllum inophyllum", 37: "Calotropis gigantea", 38: "Camellia chrysantha",
    39: "Caprifoliaceae", 40: "Capsicum annuum", 41: "Carica papaya",
    42: "Catharanthus roseus", 43: "Celastrus hindsii", 44: "Celosia argentea",
    45: "Centella asiatica", 46: "Citrus aurantifolia", 47: "Citrus hystrix",
    48: "Clausena indica", 49: "Cleistocalyx operculatus"
}

# Directory containing models
batch_model_files = [
    r"E:\lance colin y\PROJECT\Annamalai final year project\xception_medicinal_plants_batch_1.h5",
    r"E:\lance colin y\PROJECT\Annamalai final year project\xception_medicinal_plants_batch_2.h5",
    r"E:\lance colin y\PROJECT\Annamalai final year project\xception_medicinal_plants_batch_3.h5",
    r"E:\lance colin y\PROJECT\Annamalai final year project\xception_medicinal_plants_batch_4.h5",
    r"E:\lance colin y\PROJECT\Annamalai final year project\xception_medicinal_plants_batch_5.h5",
    r"E:\lance colin y\PROJECT\Annamalai final year project\Model_Mobilenet.h5"
]

# Function to load and preprocess image
def load_and_preprocess_image(image_path, image_size=(299, 299)):
    if not os.path.isfile(image_path):
        print(f"Error: Image file does not exist at {image_path}")
        return None
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to load and preprocess test data
def load_test_images(image_dir, class_mapping, image_size=(299, 299)):
    images, labels = [], []
    for label_dir in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label_dir)
        if os.path.isdir(label_path) and label_dir in class_mapping.values():
            label = list(class_mapping.keys())[list(class_mapping.values()).index(label_dir)]
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                try:
                    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    print(f"Loaded {len(images)} test images.")
    return np.array(images), np.array(labels)

# Function to predict with the best model
def predict_with_best_model(image_path, model_files, class_mapping):
    best_model = None
    best_accuracy = 0
    best_roc_auc = -1

    image_data = load_and_preprocess_image(image_path)
    if image_data is None:
        return None

    for model_path in model_files:
        model = load_model(model_path)
        predictions = model.predict(image_data)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Convert predicted class to one-hot encoding for y_true
        y_true = np.zeros_like(predictions[0])
        y_true[predicted_class] = 1

        # Calculate accuracy and ROC AUC
        y_pred = np.argmax(predictions, axis=1)
        accuracy = accuracy_score([predicted_class], y_pred)
        fpr, tpr, _ = roc_curve(y_true, predictions[0])  # Use y_true in one-hot format
        roc_auc = auc(fpr, tpr)

        if accuracy > best_accuracy and roc_auc > best_roc_auc:
            best_accuracy = accuracy
            best_roc_auc = roc_auc
            best_model = model_path
            best_predicted_class = predicted_class

    print(f"Best Model: {best_model}")
    print(f"Predicted Class: {class_mapping[best_predicted_class]}")
    return class_mapping[best_predicted_class]

# ROC Curve plotting
def plot_roc_curve(predictions, true_labels, num_classes, class_mapping):
    fpr = {}
    tpr = {}
    roc_auc = {}

    true_labels_one_hot = tf.keras.utils.to_categorical(true_labels, num_classes=num_classes)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_one_hot[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} ({class_mapping[i]}) (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Example of using single image
    image_path = r"E:\lance colin y\PROJECT\Annamalai final year project\Dataset\train\Batch_1\Aloe vera\6.JPG"
    predicted_class = predict_with_best_model(image_path, batch_model_files, class_mapping)

    # If you want to evaluate all models on the test data
    test_image_dir = r"E:\lance colin y\PROJECT\Annamalai final year project\Dataset\test_new"
    test_images, test_labels = load_test_images(test_image_dir, class_mapping)

    # Normalize test images
    test_images = test_images / 255.0
    num_classes = len(class_mapping)

    # Evaluate each batch model
    for batch_index, model_path in enumerate(batch_model_files, start=1):
        print(f"\nTesting Batch {batch_index}: {model_path}")
        model = load_model(model_path)

        # Predict on the test data
        y_pred_proba = model.predict(test_images)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Compute accuracy and classification report
        print("Accuracy:", accuracy_score(test_labels, y_pred))
        print("Classification Report:\n", classification_report(test_labels, y_pred, target_names=list(class_mapping.values())))

        # Plot ROC curve
        plot_roc_curve(y_pred_proba, test_labels, num_classes, class_mapping)

#!/usr/bin/env python
# coding: utf-8

# In[21]:


# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install pandas')
# get_ipython().system('pip install scikit-learn')


# In[19]:


import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pandas as pd
import time
from tensorflow.keras.applications import DenseNet169, InceptionV3, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, matthews_corrcoef
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization


# In[2]:


def center_crop(image, target_size=(256, 256)):
    h, w, _ = image.shape
    start_x = (w - target_size[0]) // 2
    start_y = (h - target_size[1]) // 2
    return image[start_y:start_y + target_size[1], start_x:start_x + target_size[0]]


# In[10]:


def process_and_augment_data(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2
    )
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        target_class_path = os.path.join(target_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image = center_crop(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            
            aug_iter = datagen.flow(image, batch_size=1)
            aug_image = next(aug_iter)[0].astype(np.uint8)
            
            save_path = os.path.join(target_class_path, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
       


# In[4]:


train_source = r"C:\Users\tanma_8x\Desktop\DL_project\DL221AI015\DL221AI015\DLai015\Dataset of groundnut plant leaf images for classification and detection\Dataset of groundnut plant leaf images for classification and detection\Groundnut_Leaf_dataset\Groundnut_Leaf_dataset\train"
test_source = r"C:\Users\tanma_8x\Desktop\DL_project\DL221AI015\DL221AI015\DLai015\Dataset of groundnut plant leaf images for classification and detection\Dataset of groundnut plant leaf images for classification and detection\Groundnut_Leaf_dataset\Groundnut_Leaf_dataset\test"
train_target = 'processed_train'
test_target = 'processed_test'


# In[5]:


process_and_augment_data(train_source, train_target)
process_and_augment_data(test_source, test_target)


# In[12]:


datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_target, target_size=(256, 256), batch_size=8, class_mode='categorical')
test_generator = datagen.flow_from_directory(test_target, target_size=(256, 256), batch_size=8, class_mode='categorical')


# In[7]:


import matplotlib.pyplot as plt
import random

# Function to load and display images
def display_images(directory, num_images=5):
    class_folders = [os.path.join(directory, c) for c in os.listdir(directory)]
    class_folders = [c for c in class_folders if os.path.isdir(c)]  # Filter only directories
    
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        class_folder = random.choice(class_folders)
        image_name = random.choice(os.listdir(class_folder))
        image_path = os.path.join(class_folder, image_name)
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        
        plt.subplot(1, num_images, i+1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(os.path.basename(class_folder))
    
    plt.show()

# Display images from the processed training dataset
display_images('processed_train', num_images=5)


# In[8]:


from tensorflow.keras.applications import DenseNet169, InceptionV3, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[9]:


# Load pretrained models
input_shape = (256, 256, 3)
densenet = DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)


# In[10]:


# Freeze base models
densenet.trainable = False
inception.trainable = False
xception.trainable = False


# In[11]:


def create_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(train_generator.num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)


# In[12]:


model_densenet = create_model(densenet)
model_inception = create_model(inception)
model_xception = create_model(xception)


# In[13]:


# Ensemble model averaging
inputs = tf.keras.Input(shape=input_shape)
outputs_densenet = model_densenet(inputs)
outputs_inception = model_inception(inputs)
outputs_xception = model_xception(inputs)
averaged_output = tf.keras.layers.Average()([outputs_densenet, outputs_inception, outputs_xception])
ensemble_model = Model(inputs=inputs, outputs=averaged_output)


# In[14]:


# Compile the model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Precision, Recall
ensemble_model.compile(
    optimizer=RMSprop(learning_rate=2.00E-05),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# Train the model
start_time = time.time()
history = ensemble_model.fit(train_generator, validation_data=test_generator, epochs=20)
training_time = time.time() - start_time


# In[15]:


# Display parameters layer-by-layer
for layer in ensemble_model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")
    for param in layer.weights:
        print(f"  - {param.name}: {param.shape}")


# In[16]:


# Save the trained model
ensemble_model.save('ensemble_model.h5')

# Print model summary and parameters
ensemble_model.summary()


# In[17]:


print(type(test_generator))


# In[18]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Load trained model
ensemble_model = tf.keras.models.load_model("ensemble_model.h5")

# List of functional sub-models
functional_models = ["functional", "functional_1", "functional_2"]

# Extract BatchNorm layers (Limit to 3 layers per sub-model)
bn_layers_dict = {}
max_bn_layers = 3 

for func_name in functional_models:
    sub_model = ensemble_model.get_layer(func_name)
    bn_layers = [layer.name for layer in sub_model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
    bn_layers_dict[func_name] = bn_layers[:max_bn_layers]  # Limit to 3 BN layers

# Ensure test_generator exists and get a batch
test_batch = next(iter(test_generator))[0]  # Extract only input images

# Create a directory to save plots
os.makedirs("bn_plots", exist_ok=True)

# Extract BN activations for Epochs 1 to 5
for epoch in range(1, 6):  # Runs for epochs 1, 2, 3, 4, 5
    print(f"\nExtracting Batch Normalization activations for Epoch {epoch}...")

    for func_name, bn_layers in bn_layers_dict.items():
        sub_model = ensemble_model.get_layer(func_name)

        for i, layer_name in enumerate(bn_layers):
            try:
                # Get Batch Normalization layer
                bn_layer = sub_model.get_layer(layer_name)

                # Temporarily unfreeze BN layer
                bn_layer.trainable = True  

                # Create an intermediate model to extract BN activations
                intermediate_model = tf.keras.Model(
                    inputs=sub_model.input,  # Correct input reference
                    outputs=bn_layer.output
                )

                # Run forward pass with `training=True` to capture BN statistics
                normalized_values = intermediate_model(test_batch, training=True)
                flattened_values = normalized_values.numpy().reshape(normalized_values.shape[0], -1)

                # Re-freeze BN layer after extraction
                bn_layer.trainable = False  

                # Plot the BN activations (LIMIT TO 3 SAMPLES MAX)
                plt.figure(figsize=(12, 6))
                for j in range(min(flattened_values.shape[0], 3)):  # Only plot 3 samples
                    plt.plot(flattened_values[j])

                plt.xlabel("Feature Index")
                plt.ylabel("Normalized Value")
                plt.title(f"Epoch {epoch} - BN Activations - {func_name} - {layer_name}")
                plt.grid(True)

                # Save the plot
                filename = f"bn_plots/epoch{epoch}_bn_{func_name}_{layer_name}.jpeg"
                plt.savefig(filename)
                plt.close()

                print(f"Saved: {filename}")

            except KeyError as e:
                print(f"Skipping {layer_name} in {func_name} due to KeyError: {e}")
            except Exception as e:
                print(f"Error processing {func_name} - {layer_name}: {e}")


# In[19]:


# Evaluate on test data
start_test_time = time.time()
y_true = test_generator.classes
y_pred_probs = ensemble_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
testing_time = time.time() - start_test_time


# In[20]:


# Compute metrics
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys(), output_dict=True)
conf_matrix = confusion_matrix(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)


# In[21]:


# Save evaluation results
results_df = pd.DataFrame(report).transpose()
results_df['MCC'] = mcc
results_df.to_excel('gagan-221ai019-prediction.xlsx', index=False)


# In[22]:


from sklearn.preprocessing import label_binarize

# Binarize the output labels for one-vs-rest ROC computation
y_true_bin = label_binarize(y_true, classes=list(range(train_generator.num_classes)))

# Compute ROC curve and AUC for each class
plt.figure(figsize=(8, 6))
for i in range(train_generator.num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

# Plot settings
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Multiclass)')
plt.legend(loc='lower right')
plt.savefig('gagan-221AI019-roc.jpeg')
plt.show()


# In[23]:


# Plot Accuracy and Loss Graphs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')
plt.savefig('gagan-221AI019-accuracygraph.jpeg')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')
plt.savefig('gagan-221AI019-lossgraph.jpeg')
plt.show()


# In[24]:


# Save training and testing time
time_results = pd.DataFrame({
    'Training Time (s)': [training_time],
    'Testing Time (s)': [testing_time]
})
time_results.to_excel('gagan-221AI019-trainingtime.xlsx', index=False)


# In[2]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install numpy as np')


# In[11]:


import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r"C:\Users\Chaitanya\Desktop\DL_project\DL221AI015\DL221AI015\DLai015\ensemble_model.h5", compile=False)

# Load class labels dynamically from train generator
class_labels = list(test_generator.class_indices.keys())  # Ensure test_generator is defined

def predict_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image) / 255.0  # Ensure same normalization as training
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    predictions = model.predict(image)  # Ensure batch norm is in inference mode
    probabilities = tf.nn.softmax(predictions).numpy()  # Convert logits to probabilities
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[0][predicted_class]
    
    print(f"Predicted Class: {class_labels[predicted_class]} (Confidence: {confidence:.2f})")

# Example usage
image_path = r"C:\Users\tanma_8x\Downloads\ed5d4d66-a04b-433f-9cc5-9d633a662574.jpg"
predict_image(image_path)


# In[ ]:





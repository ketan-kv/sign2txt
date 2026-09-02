import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import json

# Configuration
DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 5  # Keeping it low for faster initial training

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Initializing Data Generators...")
    # Rescaling and 20% validation split
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    
    print("Loading Training Data...")
    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    print("Loading Validation Data...")
    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Save the class mapping for the application to use
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
        
    print(f"Classes detected: {len(class_names)}")
        
    model = build_model(len(class_indices))
    model.summary()
    
    print(f"Starting Training for {EPOCHS} epochs...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )
    
    model.save("isl_model.h5")
    print("Training complete! Model saved to isl_model.h5")

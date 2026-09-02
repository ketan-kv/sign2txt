import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATASET_DIR = "dataset"
IMG_SIZE = 224
BATCH_SIZE = 32

def main():
    if not os.path.exists(DATASET_DIR):
        print("Error: dataset/ folder not found.")
        return

    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    num_classes = len(classes)
    print(f"Found {num_classes} classes.")

    # Save class names mapping for app.py
    label_map = {str(i): cls for i, cls in enumerate(classes)}
    with open("labels.json", "w") as f:
        json.dump(label_map, f)

    # Heavy Data Augmentation to bridge the domain gap (black bg -> webcam)
    # Brightness and contrast shifts are critical here.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.7, 1.3],
        horizontal_flip=False, # Do NOT flip ISL signs, handedness matters
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Transfer Learning: MobileNetV2 Base (Pre-trained on ImageNet)
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    )
    
    # Freeze the base model to prevent destroying pre-trained weights initially
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    print("\nPhase 1: Training the custom head (Base frozen)...")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=callbacks
    )

    print("\nPhase 2: Fine-tuning top layers of MobileNetV2...")
    # Unfreeze the top layers of MobileNetV2 for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]: # Keep early feature extractors frozen
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Tiny learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=callbacks
    )

    model.save("isl_cnn_model.keras")
    print("\nTraining complete! Saved to isl_cnn_model.keras")

if __name__ == "__main__":
    main()

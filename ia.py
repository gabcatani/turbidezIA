import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Parâmetros
img_size = (150, 150)
batch_size = 32

# Data Augmentation e pré-processamento
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # para dividir o conjunto de treinamento em treinamento e validação
)

# Generadores de dados
train_gen = train_datagen.flow_from_directory(
    directory='./fotos2',  # diretório raiz onde as subpastas de classes estão localizadas
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # use para treinamento
)

val_gen = train_datagen.flow_from_directory(
    directory='./fotos2',  # diretório raiz onde as subpastas de classes estão localizadas
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # use para validação
)
# Modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_gen.class_indices), activation='softmax')
])

# Compilação
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Treinamento
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    batch_size=batch_size,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Salvar modelo
model.save("turbidez_model.keras")

# Salvar o dicionário de classes
import json
with open('class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

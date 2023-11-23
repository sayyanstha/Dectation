import TensorFlow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into training and validation sets
)

batch_size = 32
image_size = (224, 224)

train_generator = datagen.flow_from_directory(
    'data_directory',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Specify training set
)

val_generator = datagen.flow_from_directory(
    'data_directory',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Specify validation set
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Evaluate the model
test_generator = datagen.flow_from_directory(
    'test_data_directory',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)

print (classification_report(test_generator.classes, y_pred))
print( confusion_matrix(test_generator.classes, y_pred))

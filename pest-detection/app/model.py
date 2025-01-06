import tensorflow as tf
from app.utils import create_data_generators, plot_training_curves

def build_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Build and compile the CNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """
    Train the model and save it as 'saved_model.h5'.
    """
    train_dir = "DataSet/Train"
    val_dir = "DataSet/Validation"
    train_gen, val_gen = create_data_generators(train_dir, val_dir)

    model = build_model(num_classes=train_gen.num_classes)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    model.save("saved_model.h5")
    print("Model saved as 'saved_model.h5'")
    plot_training_curves(history)

if __name__ == "__main__":
    train_model()

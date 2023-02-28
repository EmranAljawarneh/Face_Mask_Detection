# Define CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile model with cross-entropy loss and Adam optimizer
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model on training set
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))


y_pred = model.predict(X_test).round()
# Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)


# find the model accuracy
print(f'Accuracy of the model: {np.round(test_acc*100,2)}%')
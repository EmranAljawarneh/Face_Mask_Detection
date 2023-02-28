# Split the data
def split_data(data, labels):
  (X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.3, random_state=0)
  X_train = X_train.astype("float32") / 255.0
  X_test = X_test.astype("float32") / 255.0

  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(data, labels)
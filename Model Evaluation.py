# Evaluate the model using external dataset
with_mask_test_path = glob.glob('/content/with_mask_test/*.png')
without_mask_test_path = glob.glob('/content/without_mask_test/*.png')
test_image_path = with_mask_test_path+without_mask_test_path

test_images = []

for test_image_paths in test_image_path:
  # load the input image (224x224) and preprocess it
  test_image = cv2.imread(test_image_paths)
  test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
  test_image = cv2.resize(test_image, (224, 224))
  test_image = img_to_array(test_image)
  test_image = preprocess_input(test_image)
  # update the data and labels lists, respectively
  test_images.append(test_image)

# convert the data and labels to NumPy arrays
test_data = np.array(test_images, dtype="float32")

# shuffle the data
random.shuffle(test_data)

# Predict the results
test = model.predict(test_data)

# Find the indices of the maximum values of an array as a one-dimensional array
predIdxs = np.argmax(test, axis=1)
print(predIdxs)

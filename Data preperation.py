# Read and prepare the data
with_mask_path = glob.glob('/content/with_mask/*.png')
without_mask_path = glob.glob('/content/without_mask/*.png')
image_path = with_mask_path+without_mask_path

label_mask = []
label_no_mask = []
labels = []
images = []

def prepare_data(image_path):
  for image_paths in image_path:
    # load the input image (224x224) and preprocess it

    image = cv2.imread(image_paths)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
  # update the data and labels lists, respectively
    images.append(image)

  # convert the data and labels to NumPy arrays
  data = np.array(images, dtype="float32")

  for i in with_mask_path:
    label_mask.append(1)    # represent that the person is wearing mask.
  for a in without_mask_path:
    label_no_mask.append(0) # represent that the person is not wearing mask.

  label = label_mask+label_no_mask
  labels = np.array(label)

  return data, labels
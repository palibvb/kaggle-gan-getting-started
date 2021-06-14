tileUp_nonVisualFeatures = True

# Generation resolution - Must be square
# Training data is also scaled to this.
# Note GENERATE_RES 4 or higher
# will blow Google CoLab's memory and have not
# been tested extensivly.
GENERATE_RES = 8 # Generation resolution factor  # 8 == 256
# (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = "/content/drive/My Drive/gan-getting-started/monet_jpg"

load_extra_monets = True
EPOCHS = 5
BATCH_SIZE = 32
BUFFER_SIZE = 60000

TRAIN_GENERATOR_ON_ONES = True
TRAIN_GENERATOR_ON_KAGGLE_PHOTOS = False
KAGGLE_PHOTOS_PATH = "/content/drive/My Drive/gan-getting-started/photo_jpg"

training_binary_path = os.path.join(DATA_PATH,  f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

# print(f"Will generate {GENERATE_SQUARE}px square images.")

# Image set has 11,682 images.  Can take over an hour
# for initial preprocessing.
# Because of this time needed, save a Numpy preprocessed file.
# Note, that file is large enough to cause problems for
# sume verisons of Pickle,
# so Numpy binary files are used.
# training_binary_path = os.path.join(DATA_PATH,  f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')
# import os
# os.chdir("/content/drive/My Drive/gan-getting-started/monet_jpg")
# !ls

def pull_data(DATA_PATH, max_samples_to_load=None):
  data = []
  monets_path = os.path.join(DATA_PATH)

  if max_samples_to_load==None:
    for filename in tqdm(os.listdir(monets_path)):

        if os.path.isfile(os.path.join(monets_path,filename)):
            path = os.path.join(monets_path,filename)

            if path.endswith('.jpg'):
              image = Image.open(path) .resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
              data.append( np.asarray( image ))
  else:
    i = 0
    for filename in tqdm(os.listdir(monets_path)):
        if i >= max_samples_to_load:
          break

        if os.path.isfile(os.path.join(monets_path,filename)):
            path = os.path.join(monets_path,filename)

            if path.endswith('.jpg'):
              image = Image.open(path) .resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
              data.append( np.asarray( image ))
        i=i+1

  # [TODO]: printe mal 20 bilder.

  return data
if not os.path.isfile(training_binary_path):
  training_data = pull_data(DATA_PATH)
  np.save(training_binary_path, training_data)
else:
  training_data = np.load(training_binary_path)
print("Data-shape direkt nach pull:", np.shape(training_data))
training_data = np.reshape( training_data, (-1, GENERATE_SQUARE,   GENERATE_SQUARE, IMAGE_CHANNELS))
training_data = training_data.astype( np.float32 )
fid_image1 = training_data
fid_image1.shape
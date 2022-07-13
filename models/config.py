BATCH_SIZE = 8
INPUT_CHANNELS = 1  # the number of channels of the images
LAYERS_DEPTH = [64, 128, 256, 512, 1024]  # number of filters at each layer in the encoder/decoder
LEARNING_RATE = 1e-3
MAX_EPOCHS = 60
TRANSITION_LENGTH = 9  # how many z-stack levels are used for each image
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.1
USE_TEN_CROP = True  # whether to generate 10 crops from each image in the training/validation set, or a center crop
USE_TEN_CROP_TESTING = False  # whether to generate 10 crops from each image in the test set, or a center crop
NORMALIZE_IMAGES = False
INPUT_CROP_SIZE = 128
MEAN_W1 = 0.14
MEAN_W2 = 0.29
STD_W1 = 0.15
STD_W2 = 0.15
DATASET_SIZE = 1536
STACKS = range(0, 17, 2)  # change accordingly
DEBLUR_INPUTS_FILE = "test_deblur"   # set this to the name of the CSV file containing the inputs for deblurring (the
# filename should be of the form `name_w1.csv` or 'name_w2.csv'. Here, the string should only be the `name` part of
# the filename.
import os                         # For operating system
import re                         # Regular expression
import cv2                        # Image processing
import glob                       # Unix style pathname pattern expansion
import random                     # For random number generation
import shutil                     # High-level file operations
import requests                   # Reading url
from PIL import Image             # Python Image Library
from bs4 import BeautifulSoup     # Web scrapping
import matplotlib.pyplot as plt   # Making plots
import matplotlib.image as mpimg  # To view color image
import wget                       # For downloading images
from tqdm import tqdm             # For progress bars
from zipfile import ZipFile       # To unzip
# %matplotlib inline

CATEGORY = ["Fire", "NoFire"]

# Base directroy where dataset is stored locally
base_dir = 'C:/Users/falli/PycharmProjects/FireDetecting/Images'
dataset_test_dir = 'C:/Users/falli/PycharmProjects/FireDetecting/Test_Dataset'
dataset_train_dir = 'C:/Users/falli/PycharmProjects/FireDetecting/Dataset'

# make train directories
train_dir = os.path.join(base_dir, 'train')
train_dir_fire = os.path.join(train_dir, 'Fire')
if not os.path.exists(train_dir_fire):
    os.makedirs(train_dir_fire)
train_dir_no_fire = os.path.join(train_dir, 'NoFire')
if not os.path.exists(train_dir_no_fire):
    os.makedirs(train_dir_no_fire)

# make test directories
test_dir = os.path.join(base_dir, 'test')
test_dir_fire = os.path.join(test_dir, 'Fire')
if not os.path.exists(test_dir_fire):
    os.makedirs(test_dir_fire)
test_dir_no_fire = os.path.join(test_dir, 'NoFire')
if not os.path.exists(test_dir_no_fire):
    os.makedirs(test_dir_no_fire)

def resize_training_set():
    # dataset = "Dataset"
    # if not os.path.exists("training_dataset"):
    #     os.makedirs("training_dataset")
    #     os.makedirs("training_dataset/Fire")
    #     os.makedirs("training_dataset/NoFire")
    for category in CATEGORY:
        path = os.path.join(dataset_train_dir, category)
        print('READING DATASET :' + category + '\n')
        for img in tqdm(os.listdir(path)):
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                if image.shape[2] == 3:
                    resized_image = cv2.resize(image, (64, 64))
                    cv2.imwrite(path + "/" + img, resized_image)
            except Exception as e:
                pass

def resize_test_set():
    # dataset = "Test_Dataset"
    # if not os.path.exists("testing_dataset"):
    #     os.makedirs("testing_dataset")
    #     os.makedirs("testing_dataset/Fire")
    #     os.makedirs("testing_dataset/NoFire")
    for category in CATEGORY:
        path = os.path.join(dataset_test_dir, category)
        print('READING DATASET FOR TEST :' + category + '\n')
        for img in tqdm(os.listdir(path)):
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                if image.shape[2] == 3:
                    resized_image = cv2.resize(image, (64, 64))
                    cv2.imwrite(path + "/" + img, resized_image)
            except Exception as e:
                pass

def blur_dataset():
    # dataset = "training_dataset"
    print("BLURRING: " + dataset_train_dir)
    for category in tqdm(CATEGORY):
        path = os.path.join(dataset_train_dir, category)
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                blur_image = cv2.blur(image, (5, 5))
                cv2.imwrite(path + "/blur_" + img, blur_image)
            except Exception as e:
                pass


def symmetry():
    # dataset = "training_dataset"
    print("REVERSE: " + dataset_train_dir)
    for category in tqdm(CATEGORY):
        path = os.path.join(dataset_train_dir, category)
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                fliped_image = cv2.flip(image, 1)
                cv2.imwrite(path + "/flip_" + img, fliped_image)
            except Exception as e:
                pass


def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))


def download_dataset():
    print("DOWNLOADING TRAINING DATASET\n")
    dataset_name = "Dataset.zip"
    url = "https://firenetdataset.s3-us-west-2.amazonaws.com/"
    wget.download(url + dataset_name, "./" + dataset_name, bar=bar_custom)

    print("DOWNLOADING TEST DATASET")
    dataset_name = "Test_Dataset.zip"
    wget.download(url + dataset_name, "./" + dataset_name, bar=bar_custom)


def inflate_dataset():
    print("UNCOMPRESS TRAINING DATASET")
    with ZipFile("./Dataset.zip", "r") as zipObj:
        tqdm(zipObj.extractall())
    print("UNCOMPRESS TEST DATASET")
    with ZipFile("./Test_Dataset.zip", "r") as zipObj:
        tqdm(zipObj.extractall())


def augment_data():
    blur_dataset()
    symmetry()


def image_collage(path, n_rows, n_cols, title):
    '''This function selects the first few images from given
    path and presents in the collage form.
    path is the directory from which images are taken.
    n_rows and n_cols are rows and column of the collage.
    title is the title of the collage.'''

    # Initiating the plot
    fig = plt.figure()
    plt.title(title)

    # Turns off axis from the collage (total plot)
    plt.axis('Off')

    # List of all images in the path
    img_list = os.listdir(path)

    img_num = 1
    for img in img_list[:n_rows * n_cols]:
        # directroies of the choosen images
        img_path = os.path.join(path, img)

        # showing the color images
        img_read = mpimg.imread(img_path)

        # collage is formed from subplot
        ax = fig.add_subplot(n_rows, n_cols, img_num)
        img_num += 1

        # displaying image in subplot
        ax.imshow(img_read)

        # turns off the axis from individual image in the collage
        ax.axis('Off')


def prepare_dataset():
    download_dataset()
    inflate_dataset()
    resize_training_set()
    resize_test_set()
    augment_data()
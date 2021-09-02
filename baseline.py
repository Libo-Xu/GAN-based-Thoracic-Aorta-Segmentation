import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import os
import numpy as np
from Unet import get_UNet
from plot import plot_learning_curve
from metrics import dice_coef_loss, dice_coef
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import model_from_json
from help_functions import *
import tensorflow as tf
tf.test.gpu_device_name()

### Read and normalize data

## Already compressed the nii image to npz file
def read_data(image_path, if_image=False):
    filename = os.listdir(image_path)
    filename.sort()

    if if_image:
        img_compressed = np.load(image_path + filename[0])
    else:
        img_compressed = np.load(image_path + filename[1])

    for i in range(40): # 50 NCE images in total, 40 for training, 10 for testing
        if i == 0:
            image = img_compressed['arr_0']
        else:
            image = np.append(image, img_compressed['arr_{}'.format(i)], axis=0)

    print(image.shape)
    print('npz files loaded.')

    return image


def read_data_predict(image_path, i, if_image=False):
    filename = os.listdir(image_path)
    filename.sort()
    if if_image:
        img_compressed = np.load(image_path + filename[0])
    else:
        img_compressed = np.load(image_path + filename[1])

    image = img_compressed['arr_{}'.format(i)]

    return image


def flatten(ndarray):
    chs = ndarray.shape[0]
    h = ndarray.shape[1]
    w = ndarray.shape[2]
    ndarray = np.reshape(ndarray, (chs * h * w))
    return ndarray

'''
## read nii images
def read_data(dir, if_image = False):

    for root, dirs, files in os.walk(dir):
        img = sitk.ReadImage(dir + files[0])
        data = sitk.GetArrayFromImage(img)
#         print(data.shape)
#         for i in range(1, 5):    
#             img = sitk.ReadImage(dir + files[i])
#             img = sitk.GetArrayFromImage(img)
#             print(img.shape)
#             data = np.append(data, img, axis=0)


    N_slice = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]
    tmp = np.zeros((N_slice,height//2,width//2))
    for i in range(N_slice):
        if if_image:
            tmp[i] = zoom(data[i], zoom = 0.5, order=1)
        else:
            tmp[i] = zoom(data[i], zoom = 0.5, order=1)


    data = np.reshape(tmp,(N_slice,height//2,width//2,1))

    if if_image:
        data = data/1000
    else:
        data = data.astype('float64')
#         data = normalize(data)
    print(data.shape)
    return data

def normalize(imgs):
    MIN_BOUND = -1000.0
    MAX_BOUND = 500.0

    imgs = (imgs - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    imgs[imgs>1] = 1.
    imgs[imgs<0] = 0.
    
#     imgs = imgs * 255.
    return imgs    
'''


### Training ###
################
img_path = './NCE/' # Non contrast-enhanced

imgs = read_data(img_path, if_image=True)
masks = read_data(img_path)

height = imgs.shape[1]
width = imgs.shape[2]
n_ch = imgs.shape[3]

model = get_UNet(img_shape=(height, width, n_ch), Base=32, depth=4, inc_rate=2,
                 activation='relu', drop=0.5, batchnorm=True, N=2)
model.compile(optimizer=Adam(lr=2e-4), loss=[dice_coef_loss],
              metrics=[dice_coef])
print("Check: final output of the network:")
print(model.output_shape)

experiment_name = 'baseline_cv5_1'
save_path = './' + experiment_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
json_string = model.to_json()
open(save_path + experiment_name + '_architecture.json', 'w').write(json_string)

History = model.fit(imgs,
                    masks,
                    epochs=75,
                    batch_size=16,
                    verbose=2,
                    shuffle=True)

weights_path = save_path + experiment_name + '_epoch100_weights.h5'
model.save_weights(weights_path)

# Plot the learning curve
plot_learning_curve(History, save_path + 'learning_curve')

### Predicting ###
##################
img_path = './NCE/'
# ori_path = './images/'
# files = os.listdir(ori_path)
# files.sort()

dice = np.zeros(10)
precision = np.zeros(10)
recall = np.zeros(10)
# ASSD = np.zeros(10)
eps = np.finfo(np.float32).eps
t = 0

# Load the model and the best weight

model_path = './baseline_model/cv1/baseline_cv5_1_epoch75_architecture.json'
weight_path = './baseline_model/cv1/baseline_cv5_1_epoch75_weights.h5'

with open(model_path, 'r') as file:
    model_json1 = file.read()
model = model_from_json(model_json1)
model.load_weights(weight_path)

ori_path = './nii_images/'
files = os.listdir(ori_path)
files.sort()
# Predict on the last 10 images
for j in range(40, 50):
    img = read_data_predict(img_path, j, if_image=True)
    mask = read_data_predict(img_path, j)
    pred = model.predict(img)
    # np.save('./baseline_model/pred/{}_baseline.npy'.format(j + 1), pred)


    ### Save array prediction to nii images
    # Details in help_functions.py
    pred_ = binary(pred)
    pred_ = upsample(pred_)
    print("Upsampled predicted images size : {}".format(pred.shape))
    template = ori_path + files[j] # original nii image
    outputfile = './baseline_model/pred_nii/{}_pred.nii.gz'.format(j + 1)
    itk_image = saveNumpyImageToITKImage(template, outputfile, pred_)


    # Calculate Average Symmetric Surface Distance (ASSD)
    ct = sitk.ReadImage(ori_path + files[j])
    pred_ = binary(pred)
    spacing = ct.GetSpacing()
    ori_mask = upsample(mask)
    ori_pred = upsample(pred_)
    ASSD[t] = get_ASSD(ori_mask, ori_pred, spacing)

    # Calculate recall, precisin and DSC
    pred = flatten(pred)
    mask = flatten(mask)
    recall[t] = sum(pred * mask + eps) / (sum(mask) + eps)
    precision[t] = sum(pred * mask + eps) / (sum(pred) + eps)
    dice[t] = 2 * sum(pred * mask + eps) / (sum(pred) + sum(mask) + eps)
    t = t + 1

### Print the metrics
print('dice: {}'.format(dice))
print('{} ± {}'.format(np.mean(dice), np.std(dice)))


print('recall: {}'.format(recall))
print('{} ± {}'.format(np.mean(recall), np.std(recall)))


print('precision: {}'.format(precision))
print('{} ± {}'.format(np.mean(precision), np.std(precision)))


print('ASSD: {}'.format(ASSD))
print('{} ± {}'.format(np.mean(ASSD), np.std(ASSD)))


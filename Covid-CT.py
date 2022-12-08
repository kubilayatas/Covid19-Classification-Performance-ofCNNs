import argparse
###############################################################################
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset which includes 'images' folder and 'metadata.csv' file")
ap.add_argument("-p", "--plot", type=str, default="_Training_Results.png",
	help="path to output loss/accuracy graphical plot file with extention '.png'")
ap.add_argument("-m", "--savemodel", type=str, default="./",
	help="path to output final trained model")
ap.add_argument("-e", "--epochs", type=int, default=50,
	help="An integer input to Epoch number")
ap.add_argument("-b", "--batchSize", type=int, default=8,
	help="An integer input to Batch Size")
ap.add_argument("-l", "--learningRate", type=float, default=1e-3,
	help="A floating point input to LearningRate, default is 1e-3")
ap.add_argument("-n", "--trainNumber", type=float, default=720,
	help="A floating point input to LearningRate, default is 1e-3")
ap.add_argument("-t", "--topmodel", type=str, default="resnet50",
	help="Input to select pretrained top model, default is 'resnet50'. mobilenet_v2, vgg16, efficientnet-b3,\
	are available.")
args = vars(ap.parse_args()) 


# USAGE
# python train.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNetV2
from efficientnet.keras import EfficientNetB3

from tensorflow.keras.layers import AveragePooling2D,Convolution2D,MaxPooling2D,Activation,concatenate,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from imutils import paths
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import nibabel as nib
import random
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils

INIT_LR = args["learningRate"]
EPOCHS = args["epochs"]
BS = args["batchSize"]
trainNumber = args["trainNumber"] #746
DataSet = args["dataset"]
SaveModelTo = args["savemodel"]
TopModel = args["topmodel"]
IMG_SIZE = 224
##################################
print("######################### {} Train Starts #################################".format(TopModel))
print("########[ARGS INFO]########")
print("TOP MODEL NAME: {}".format(TopModel))
print("DATASET PATH: {}".format(DataSet))
print("MODEL SAVE PATH: {}".format(SaveModelTo))
print("EPOCHS: {}".format(EPOCHS))
print("BATCH SIZE: {}".format(BS))
print("INIT_LR: {}".format(INIT_LR))
print("trainNumber: {}".format(trainNumber))
print("######[END ARGS INFO]#######")
##################################
img = nib.load(os.path.join(DataSet,"rp_im/1.nii.gz")).get_data()
covid_msk = nib.load(os.path.join(DataSet,"rp_msk/1.nii.gz")).get_data()

for i in range(2,10):
  img = np.dstack((img,nib.load(os.path.join(DataSet,"rp_im/"+str(i)+".nii.gz")).get_data()))
  covid_msk = np.dstack((covid_msk,nib.load(os.path.join(DataSet,"rp_msk/"+str(i)+".nii.gz")).get_data()))

print(img.shape, covid_msk.shape)

img -= np.full(img.shape,np.amin(img))
img = img / np.amax(img)

img = resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True)

# X_train = np.zeros((trainNumber, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
# X_test = np.zeros((img.shape[2]-trainNumber, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
data = np.zeros((trainNumber, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

print(np.amax(img),np.amin(img),img.shape,np.amax(covid_msk),np.amin(covid_msk))
img2 = np.zeros((IMG_SIZE,IMG_SIZE,3))
labels = []
sayac_normal = 0
sayac_covid = 0
sınır = trainNumber/2

i = 0
n = 0

# for i in range(covid_msk.shape[2]-29):
while (sayac_covid < sınır):
  if (np.amax(covid_msk[:,:,i]) > 0 and sayac_covid < sınır):
    labels.append("covid")
    sayac_covid += 1

    img2[:,:,0:1] = img[:,:,i:i+1]
    img2[:,:,1:2] = img[:,:,i:i+1]
    img2[:,:,2:3] = img[:,:,i:i+1]

    data[n] = img2
    n = n + 1
  
  i = i + 1

i = 0

while (sayac_normal < sınır):
  if (np.amax(covid_msk[:,:,i]) == 0 and sayac_normal < sınır):
    labels.append("normal")
    sayac_normal +=1

    img2[:,:,0:1] = img[:,:,i:i+1]
    img2[:,:,1:2] = img[:,:,i:i+1]
    img2[:,:,2:3] = img[:,:,i:i+1]

    data[n] = img2
    n = n + 1
  
  i = i + 1

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")
	
################## Models ###################

########### MobileNetv2 ################
if TopModel == "mobilenet_v2":
	baseModel = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMG_SIZE,IMG_SIZE,3))
	
	# baseModel = SqueezeNet(include_top=False, weights=None,
	#                input_tensor=None, input_shape=(IMG_SIZE,IMG_SIZE,3),
	#                pooling=None,
	#                classes=1000)
	
	# basemodel = SqueezeNet(nb_classes=1000, inputs=(3, IMG_SIZE, IMG_SIZE))
	
	headModel = baseModel.output
	headModel = GlobalAveragePooling2D()(headModel)
	headModel = Flatten(name="flatten")(headModel)
	# headModel = Dense(256, activation="relu")(headModel)
	# headModel = Dropout(0.5)(headModel)
	headModel = Dense(64, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)
	
	model = Model(inputs=baseModel.input, outputs=headModel) 
	
	for layer in baseModel.layers:
		layer.trainable = True
	
	# compile our model
	print("[INFO] compiling model...")
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	
	# train the head of the network
	print("[INFO] training head...")
	H = model.fit(					#model.fit_generator
		trainAug.flow(trainX, trainY, batch_size=BS),
		steps_per_epoch=len(trainX) // BS,
		validation_data=(testX, testY),
		validation_steps=len(testX) // BS,
		epochs=EPOCHS)

############ VGG 16 ##############
elif TopModel == "vgg16":
	baseModel = VGG16(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
	
	# construct the head of the model that will be placed on top of the
	# the base model
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(64, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)
	
	# place the head FC model on top of the base model (this will become
	# the actual model we will train)
	model = Model(inputs=baseModel.input, outputs=headModel)
	
	# loop over all layers in the base model and freeze them so they will
	# *not* be updated during the first training process
	for layer in baseModel.layers:
		layer.trainable = False
	# compile our model
	print("[INFO] compiling model...")
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	
	# train the head of the network
	print("[INFO] training head...")
	H = model.fit_generator(
		trainAug.flow(trainX, trainY, batch_size=BS),
		steps_per_epoch=len(trainX) // BS,
		validation_data=(testX, testY),
		validation_steps=len(testX) // BS,
		epochs=EPOCHS)

############## EfficientNetB3 ###################		
elif TopModel == "efficientnet-b3":
	eff_net = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)) #input_shape=(256, 256, 3)
	
	x = eff_net.output
	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	x = Dropout(0.5)(x)
	predictions = Dense(2, activation="sigmoid")(x)
	model = Model(eff_net.input, predictions)
	
	for layer in eff_net.layers:
		layer.trainable = False
	
	# model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6),loss='binary_crossentropy',metrics=['accuracy'])
	print("[INFO] compiling model...")
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	
	# train the head of the network
	print("[INFO] training head...")	
	H = model.fit_generator(
		trainAug.flow(trainX, trainY, batch_size=BS),
		steps_per_epoch=len(trainX) // BS,
		validation_data=(testX, testY),
		validation_steps=len(testX) // BS,
		epochs=EPOCHS)

############ ResNet-50 #########################3
else:
	#resnet50
	baseModel = ResNet50(weights="imagenet", include_top=False,
		input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
	
	# construct the head of the model that will be placed on top of the
	# the base model
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	# headModel = Dense(256, activation="relu")(headModel)
	# headModel = Dropout(0.5)(headModel)
	headModel = Dense(64, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)
	
	# place the head FC model on top of the base model (this will become
	# the actual model we will train)
	model = Model(inputs=baseModel.input, outputs=headModel)
	
	# loop over all layers in the base model and freeze them so they will
	# *not* be updated during the first training process
	for layer in baseModel.layers:
		layer.trainable = False

	# compile our model
	print("[INFO] compiling model...")
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])
	
	# train the head of the network
	print("[INFO] training head...")
	H = model.fit_generator(
		trainAug.flow(trainX, trainY, batch_size=BS),
		steps_per_epoch=len(trainX) // BS,
		validation_data=(testX, testY),
		validation_steps=len(testX) // BS,
		epochs=EPOCHS)
	
#############################################


#############################################
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# Saving History of train and test
pattth = os.path.sep.join([SaveModelTo,TopModel])
os.makedirs(pattth, exist_ok=True)
pattth = os.path.join(pattth,TopModel+"_TrainHistory")
print("[INFO] Saving history to: {}".format(pattth))
text_file = open(pattth, "w")
text_file.write(str(H.history))
text_file.close()
	
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig(args["plot"])
pattth = os.path.sep.join([SaveModelTo,TopModel])
pattth = os.path.join(pattth,TopModel+"_historyfigure.svg")
plt.savefig(pattth,format='svg', dpi=1200)
# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
pattth = os.path.sep.join([SaveModelTo,TopModel])
model.save(pattth+"/"+TopModel+"+_model.h5")
print("######################### {} Train Ends #################################".format(TopModel))
print("####################################################################")
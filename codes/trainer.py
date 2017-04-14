
#USAGE
# python trainer.py --dataset dataset/images/ --model models/svm.cpickle


import numpy as np
import cv2
#import serial
import time
import argparse
import glob
#from sklearn.neural_network import MLPClassifier
import cPickle
from pyimagesearch.hog import HOG
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="ruta al dataset de imagenes")
#ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
ap.add_argument("-m", "--model", required = True, help = "path to where the model will be stored")

args = vars(ap.parse_args())
directorio = args["images"]
#directorio = args["dataset"]

imagePaths = sorted(glob.glob(directorio + "/*.png"))

data = []
target = []


hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1, 1), normalize = True)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blured = cv2.GaussianBlur(gray, (3,3), 0)

	hist = hog.describe(blured)
	data.append(hist)

	label = imagePath.split("_")[-2]
	#label = label[len(directorio)+1:]
	target.append(label)
	"""
	cv2.imshow("gray", blured)

	time.sleep(0.025)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

	#data.append(features)
	
	print(imagePath)
	print(label)
	"""

# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# construct the training and testing splits
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
	test_size = 0.3, random_state = 42)

# train the model
model = LinearSVC(random_state = 42)
model.fit(data, target)

# evaluate the classifier
print classification_report(testTarget, model.predict(testData), target_names = targetNames)

#clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
#clf.fit(data, label)



# dump the model to file
f = open(args["model"], "w")
f.write(cPickle.dumps(model))
f.close()


# USAGE
# python classify_by_video.py --model models/svm.cpickle --dataset dataset/images --test_images dataset/test_images  --video video/video1.avi

# import the necessary packages
from pyimagesearch.hog import HOG
import argparse
import cPickle
import mahotas
import cv2
import glob
from sklearn.preprocessing import LabelEncoder
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "path to where the model will be stored")
ap.add_argument("-i", "--test_images", required = True, help = "path to the test images folder")
ap.add_argument("-d", "--dataset", required = True, help = "path to the dataset folder")
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())


# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(1)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])



######################## CARGANDO MODELO ###########################
print "Cargando Modelo........."
# load the model
model = open(args["model"]).read()
model = cPickle.loads(model)

########## CARGANDO IMAGENES DEL TRAINING PARA OBTENER LAS ETIQUETAS..... ########
 
print "Cargando Labels de las imagenes de Entrenamiento..........."
imagenes_entrenamiento = sorted(glob.glob(args["dataset"] + "/*.png"))
#print imagenes_entrenamiento
target = []
print "Cargando"
# loop over the image and mask paths
for imagen_entrenamiento in imagenes_entrenamiento:
	# load the image and mask
	image = cv2.imread(imagen_entrenamiento)
	target.append(imagen_entrenamiento.split("_")[-2])
	#print imagen_entrenamiento.split("_")[-2]
	#print "."

le = LabelEncoder()
target = le.fit_transform(target)

#################################################################################


# initialize the HOG descriptor
hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1, 1), normalize = True)

"""
# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur the image, find edges, and then find contours along
# the edged regions
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# extract features from the image and classify it
hist = hog.describe(blurred)
direccion = le.inverse_transform(model.predict(hist))[0]
#le.inverse_transform(model.predict(features))[0]
print " Por favor: %s" % (direccion)

cv2.putText(image, str(direccion), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
cv2.imshow("image", image)
cv2.waitKey(0)
"""


# keep looping
while True:
	# grab the current frame
	(grabbed, image) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blur the image, find edges, and then find contours along
	# the edged regions
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	# extract features from the image and classify it
	hist = hog.describe(blurred)
	direccion = le.inverse_transform(model.predict(hist))[0]
	#le.inverse_transform(model.predict(features))[0]
	print " Por favor: %s" % (direccion)

	cv2.putText(image, str(direccion), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
	cv2.imshow("image", image)
	
	thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	T = mahotas.thresholding.otsu(thresh)
	thresh[thresh > T] = 255
	thresh[thresh < T] = 0
	thresh = cv2.bitwise_not(thresh)

	# show our threshold image
	cv2.imshow("Road", thresh)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()



"""
imagePaths = sorted(glob.glob(args["test_images"] + "/*.png"))


# loop over a sample of the images
for i in np.random.choice(np.arange(0, len(imagePaths)), 18):
	# grab the image and mask paths
	imagePath = imagePaths[i]
	# load the image and convert it to grayscale
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# blur the image, find edges, and then find contours along
	# the edged regions
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	# extract features from the image and classify it
	hist = hog.describe(blurred)
	direccion = le.inverse_transform(model.predict(hist))[0]
	#le.inverse_transform(model.predict(features))[0]
	print " Por favor: %s" % (direccion)

	cv2.putText(image, str(direccion), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
	cv2.imshow("image", image)
	cv2.waitKey(0)

"""

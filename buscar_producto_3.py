# -*- coding: utf-8 -*-
# USAGE
# python buscar_producto_3.py --db productos.csv --empaques empaque 
# python buscar_producto_3.py --db productos.csv --empaques empaque --video 
# import the necessary packages
from pyimagesearch.coverdescriptor import CoverDescriptor
from pyimagesearch.covermatcher import CoverMatcher
import argparse
import glob
import csv
import cv2
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True, help = "ruta a donde se encuentra la base de datos de productos")
ap.add_argument("-c", "--empaques", required = True, help = "ruta a donde se encuentra nuestros productos ")
#ap.add_argument("-q", "--query", required = True, help = "ruta a donde se encuentra la imagen del ṕroducto a reconocer")
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

# initialize the database dictionary of covers
db = {}

# loop over the database, CSV file is opened and each line looped over.
# The db dictionary is updated with the unique filename of the book as the key and 
# the title of the book and author as the value.
for l in csv.reader(open(args["db"])):
	# update the database using the image ID as the key
	db[l[0]] = l[1:]

# initialize the cover descriptor and cover matcher
cd = CoverDescriptor()
cv = CoverMatcher(cd, glob.glob(args["empaques"] + "/*.png"))


def click_search(event, x, y, flags, param):

	if event == cv2.EVENT_LBUTTONDOWN:
		print "Buscando .... "
	elif event == cv2.EVENT_LBUTTONUP:

		# grab the current frame
		(grabbed, searched_image) = camera.read()

		# if we are viewing a video and we did not grab a
		# frame, then we have reached the end of the video
		if args.get("video") and not grabbed:
			print "No se pudo abrir el video"
		
		gray = cv2.cvtColor(searched_image, cv2.COLOR_BGR2GRAY)
		# extracts his keypoints and local invariant descriptors from the query image
		(queryKps, queryDescs) = cd.describe(gray)

		# try to match the book cover to a known database of images, A sorted list
		# of results is returned, with the best book cover matches placed at the top of the list.
		results = cv.search(queryKps, queryDescs)

		# show the query cover
		#queryImage = imutils.resize(queryImage, width = 500)
		#cv2.imshow("Query", queryImage)

		# check to see if no results were found
		if len(results) == 0:
			print "No Existe el archivo!"
			cv2.waitKey(0)

		# otherwise, matches were found
		else:

			resultado = results[0]
			(score, coverPath) = resultado
			#nombre = coverPath

			(author, title) = db[coverPath[coverPath.rfind("/") + 1:]]
			print "%d. %.2f%% : %s - %s" % (1, score * 100, author, title)
			# load the result image and show it
			result = cv2.imread(coverPath)
			result = imutils.resize(result, width = 500)
			cv2.imshow("Result", result)
			cv2.putText(result, str(title), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (155, 155, 155), 2)
		



#########################################################################################################

# keep looping
while True:
	cv2.namedWindow("buscador")
	cv2.setMouseCallback("buscador", click_search)
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	cv2.imshow("buscador", frame)
	
	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
########################################################################################################
	

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

	

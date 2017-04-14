# -*- coding: utf-8 -*-
# USAGE
# python buscar_producto.py --db productos.csv --empaques empaque --query queries/imagen1.JPG

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
ap.add_argument("-q", "--query", required = True, help = "ruta a donde se encuentra la imagen del ṕroducto a reconocer")

args = vars(ap.parse_args())

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

# load the query image, convert it to grayscale, and extract
# keypoints and descriptors
queryImage = cv2.imread(args["query"])
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
# extracts his keypoints and local invariant descriptors from the query image
(queryKps, queryDescs) = cd.describe(gray)

# try to match the book cover to a known database of images, A sorted list
# of results is returned, with the best book cover matches placed at the top of the list.
results = cv.search(queryKps, queryDescs)

# show the query cover
queryImage = imutils.resize(queryImage, width = 500)
cv2.imshow("Query", queryImage)

# check to see if no results were found
if len(results) == 0:
	print "I could not find a match for that cover!"
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

	cv2.waitKey(0)
	
	"""
	# loop over the results
	for (i, (score, coverPath)) in enumerate(results):
		# grab the book information
		(author, title) = db[coverPath[coverPath.rfind("/") + 1:]]
		print "%d. %.2f%% : %s - %s" % (i + 1, score * 100, author, title)

		if score * 100 > 65.0:
			# load the result image and show it
			result = cv2.imread(coverPath)
			result = imutils.resize(result, width = 500)
			cv2.imshow("Result", result)
		else:
			print "No se pudo encontrar"		
	"""	


	
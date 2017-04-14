# -*- coding: utf-8 -*-

# USAGE
# python convert_all_directions.py --images datos3/F --name F --initialnumber 0 --format png

"""
BY: LUIS LLANOS

Este script transforma un conjunto de imagenes de una persona nueva (las cuales deben estar en una carpeta)
les pone el nombre de la persona, la expresion facial, el numero y ell formato de salida
"""

import mahotas
import cv2
import argparse
import glob
import time
from pyimagesearch import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "path to the image dataset")
ap.add_argument("-n", "--name", required = True, help = "Nombre de la direccion (F, B, L, R, DL, DR)");
ap.add_argument("-in", "--initialnumber", required = True, help = "numero inicial para el nombre de las imagenes");
ap.add_argument("-f", "--format", required = True, help = "Nombre del formato de salida");
args = vars(ap.parse_args())


TempimagePaths = sorted(glob.glob(args["images"] + "/*.png"))
nombre_imagen = args["name"]
initialnumber = args["initialnumber"]
formato_imagen = args["format"]

print " ############# Procesando Imagenes #################"
print ""
print "Exportando imagenes ....."
"""
************* PROCESANDO IMAGENES ****************
"""
number = int(initialnumber)

for TempimagePath in TempimagePaths:
	#name_image = "image"
	image_image = cv2.imread(TempimagePath)	
	#image_image = imutils.resize(image_image, height = 800)
	#image_image = imutils.rotate(image_image, 270)
	#image_image = imutils.resize(image_image, width = 500)
	name_formal_image = "{0}_{1}.{2}".format(nombre_imagen, number, formato_imagen)
	cv2.imwrite(name_formal_image, image_image)
	number = number + 1

print "WELL DONE"
"""
print "Wait Please......................................................"
time.sleep(10)



imagePaths = sorted(glob.glob("*.png"))
print "WELL DONE"
"""

"""
print " ############ Procesando Mascaras #################"
print ""
print "Exportando Mascaras ......."
"""
"""
************* PROCESANDO MASCARAS ****************
"""
"""
for imagePath in imagePaths:
	#name_mask = "mask"
	image_mask = cv2.imread(imagePath)
	thresh = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
	T = mahotas.thresholding.otsu(thresh)
	thresh[thresh > T] = 255
	thresh[thresh < T] = 0
	thresh = cv2.bitwise_not(thresh)
	image_name = imagePath[imagePath.rfind("/")+1:]
	first = image_name.split("_")[-2]
	second = image_name.split("_")[-1]
	name_formal = "{0}_{1}_{2}".format("mask", first, second)
	cv2.imwrite(name_formal, thresh)
	
print "WELL DONE"
"""

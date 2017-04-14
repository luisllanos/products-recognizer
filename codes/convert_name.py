# -*- coding: utf-8 -*-

# USAGE
# python convert_name.py --images dataset/images/ 

"""
BY: LUIS LLANOS

Este script transforma un conjunto de imagenes de frutas nuevo (las cuales deben estar en una carpeta)
(NUEVA FRUTA) les pone tipo de imagen (IMAGEN O MASK), NOMBRE DE LA FRUTA (BANANO, PERA), 
Y EL NUMERO CORRESPONDIENTE, LUEGO se les cambia la enxtension de jpeg a png. Despues se les saca
la mascara mediante thresholding sobre el grupo de images N, se utiliza el algoritmo otsu pasa 
hallar el valor T para hacer el thresholding.Ademas pone nombres consecutivos del 0 en 
adelante algo como esto: imagen_0 

"""

import mahotas
import cv2
import argparse
import glob
import time
from pyimagesearch import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required = True, help = "path to the image dataset") 
args = vars(ap.parse_args())

imagePaths = sorted(glob.glob(args["images"] + "/*.JPG"))

print " ############# Procesando Imagenes #################"
print ""
print "Exportando imagenes ....."
"""
************* PROCESANDO IMAGENES ****************
"""
number = 0
#number = int(initialnumber)

for imagePath in imagePaths:
	first_part = "producto"
	#image_name = imagePath[imagePath.rfind("/")+1:]
	#first = image_name.split("/")[-1].split("_")[-2]
	#second = image_name.split("/")[-1].split("_")[-1]
	
	image_image = cv2.imread(imagePath)	
	#image_image = imutils.resize(image_image, height = 650)
	name_formal_image = "{0}_{1}.png".format(first_part, number)
	#cv2.imwrite(name_formal, image)
	#number = number + 1

	#nombre_formal_destino = "{0}/{1}_{2}_{3}.png".format(imagesfolder, name, nombre_imagen, number)
	#cv2.imwrite(name_formal, image)
	cv2.imwrite(name_formal_image, image_image)
	number = number + 1



# -*- coding: utf-8 -*-

# import the necessary packages
import numpy as np
import cv2

"""
SIFT and SURF produce real-valued feature vectors whereas ORB, BRIEF, and BRISK pro-
duce binary feature vectors.
SIFT or SURF use the Euclidean distance.
ORB, BRIEF, and BRISK use the Hamming distance because they pro-duce binary feature vectors
"""
# encapsulates finding keypoints in an image and then describing the area 
# surrounding each keypoint using local invariant descriptors.
# Scale Invariant Feature Descriptor (SIFT), SIFT (David Lowe’s) is used to
# keypoint detection an it uses  Difference of Gaus-sian method.
class CoverDescriptor:
	# The kpMethod is the keypoint detection method it could also be (SURF and ORB) descriptors
	# descMethod defines how the area surrounding each keypoint is described.
	# other local invariant descrip-tors can be supplied, such as SURF, ORB, BRIEF, and FREAK,
	def __init__(self, kpMethod = "SIFT", descMethod = "SIFT"):
		# store the keypoint detection method and descriptor method
		self.kpMethod = kpMethod
		self.descMethod = descMethod
	# keypoints and descriptors should be computed for the image
	def describe(self, image):
		# defines the keypoint detector 
		detector = cv2.FeatureDetector_create(self.kpMethod)
		# detect keypoints in the image
		kps = detector.detect(image)
		# extract local invariant descriptors from each keypoint,		
		extractor = cv2.DescriptorExtractor_create(self.descMethod)
		# The list of keypoints contain multiple KeyPoint objects. 
		# These objects contain infor-mation such as the (x, y) location of the keypoint, 
		# the size of the keypoint, the rotation angle, amongst other attributes.
		(kps, descs) = extractor.compute(image, kps)
		# convert the keypoints to a numpy array, he (x, y) coordi-nates of the keypoint, 
		# contained in the pt attribute.
		kps = np.float32([kp.pt for kp in kps])
		# return a tuple of keypoints and descriptors
		return (kps, descs)
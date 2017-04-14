# -*- coding: utf-8 -*-

# import the necessary packages
import numpy as np
import cv2

class CoverMatcher:
	# The descriptor is an instance of the CoverDescriptor, and the path to the 
	# di-rectory where the cover images are stored.
	def __init__(self, descriptor, coverPaths):
		# store the descriptor and book cover paths
		self.descriptor = descriptor
		self.coverPaths = coverPaths

	# the keypoints and descriptors will be matched, take the keypoints and descriptors from the 
	# query image and then match them against a database of keypoints and descriptors.
	# the best “match” will be chosen as the identification of the book cover
	def search(self, queryKps, queryDescs):
		# initialize the dictionary of results, The key of the dictio-nary will be 
		# the unique book cover filename and the value will be the matching percentage 
		# of keypoints
		results = {}
		# loop over the book cover images
		for coverPath in self.coverPaths:
			# load the query image, convert it to grayscale, and extract keypoints and 
			# descriptors. load book cover from disk, converte to grayscale and then
			# keypoints and descriptors are extracted from it using the CoverDescriptor
			cover = cv2.imread(coverPath)
			gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
			(kps, descs) = self.descriptor.describe(gray)

			# determine the number of matched, inlier keypoints,
			# then update the results
			score = self.match(queryKps, queryDescs, kps, descs)
			results[coverPath] = score

		# if matches were found, sort them
		if len(results) > 0:
			# sorted in descending order, with book covers with more keypoint
			# matches placed at the top of the list.
			results = sorted([(v, k) for (k, v) in results.items() if v > 0],
				reverse = True)

		# return the results
		return results

	# matched the query and cover database key-points and descriptors
	"""
	*kpsA: The list of keypoints associated with the first image to be matched.
	*featuresA: The list of feature vectors associated with the first image to be matched.
	*kpsB: The list of keypoints associated with the second image to be matched.
	*featuresB: The list of feature vectors associated with the second image to be matched.
	*ratio: The ratio of nearest neighbor distances sug-gested by Lowe to prune down 
	the number of key-points a homography needs to be computed for.
	*minMatches: The minimum number of matches re-quired for a homography to be calculated.
	"""
	def match(self, kpsA, featuresA, kpsB, featuresB, ratio = 0.7, minMatches = 50):
		# compute the raw matches and initialize the list of actual matches
		# BruteForce, indicating that he is going to com-pare every descriptor in featuresA
		# to every descriptor in featuresB using the euclidean distance and taking the fea-
		# ture vectors with the smallest distance as the “match”.
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		# knnMatch function of matcher. The “kNN” portion of the function stands 
		# for “k-Nearest-Neighbor”, where the “near-est neighbors” are defined by the 
		# smallest euclidean dis-tance between feature vectors. The two feature vectors 
		# with the smallest euclidean distance are considered to be “neigh-bors”. 
		# Both featuresA and featuresB are passed into the knnMatch function, 
		# with a third parameter of 2, indicating that we want to find the two 
		# nearest neighbors for each feature vector.
		rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each other, the distance
			# between the first match is less the distance of the second
			# match, times the supplied ratio.
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				# list is up-dated with a tuple of the index of the first keypoint 
				# and the index of the second keypoint.
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# check to see if there are enough matches to process, If there are not enough matches,
		# it is not worth computing the homography since the two
		# images (likely) do not contain the same book cover.
		if len(matches) > minMatches:
			# construct the two sets of points, to store the (x, y) coordinates for the for 
			# each set of matched keypoints.
			ptsA = np.float32([kpsA[i] for (i, _) in matches])
			ptsB = np.float32([kpsB[j] for (_, j) in matches])
			# compute the homography between the two sets of points an d compute the ratio of 
			# matched points (homography, which is a mapping between the two keypoint 
			# planes with the same center of projection.
			# will take his matches and de-termine which keypoints are indeed a “match” and 
			# which ones are false positives. (RANSAC algorithm, which stands for Ran-dom 
			# Sample Consensus.) RANSAC randomly samples from a set of po-tential matches. 
			# In this case, RANSAC randomly samples from Gregory’s matches list. 
			# Then, RANSAC attempts to match these samples together and verifies the hypothesis
			# of whether or not the keypoints are inliers. RANSAC con-tinues to do this until 
			# a large enough set of matches are considered to be inliers. From there, 
			# RANSAC takes the set of inliers and looks for more matches.
 
			# four arguments. The first two are ptsA and ptsB, the (x, y) coordinates of 
			# the po-tential matches The third argument is the homography method, 
			# could have used cv2.LMEDS, which is the Least-Median robust method.
			# final parameter is the RANSAC re-projection thresh-old, indicate that an error 
			# of 4.0 pixels will be tolerated for any pair of keypoints to be considered 
			# an inlier. cv2.findHomograpy function returns a tuple of two values. 
			# The first is the transformation matrix, and The status variable is a list of 
			# booleans, with a value of 1 if the corresponding keypoints in ptsA and ptsB
			# were matched, and a value of 0 if they were not.
 			(_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
			# return the ratio of the number of matched keypoints to the total number of 
			# keypoints, computes the ratio of the number of inliers to the total number 
			# of potential matches
			return float(status.sum()) / status.size

		# no matches were found, returns a value of -1.0 indicating that the
		# number of inliers could not be computed.
		return -1.0
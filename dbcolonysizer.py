# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import csv
import imghdr
import pandas as pd
import os.path
from os.path import join
from glob import glob


""" PlateMeasurer performs measurements on images of plates."""


class DBColonySizer(object):

    def __init__(self, template_image="./example_data/kernel.PNG",
                 show_images=None):
        self.kernel_file = template_image
        self.NUM_COLS = 24
        self.NUM_ROWS = 16
        self.SHOW_IMAGES = show_images  # can set to 'all' or 'missing'

    """ kernel_crop takes an image, locates the part of the image which is most similar to the kernel, and crops to that part of the image. That main idea is for this to be used to crop to just the plate. """

    def kernel_crop(self, image):
        kernel = cv2.imread(self.kernel_file, cv2.IMREAD_GRAYSCALE)
        filtered = cv2.matchTemplate(
            image=image,
            templ=kernel,
            method=cv2.TM_CCORR_NORMED)
        minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(filtered)
        cv2.rectangle(img=image, pt1=(maxLoc[0], maxLoc[1]), pt2=(
            maxLoc[0] + kernel.shape[1], maxLoc[1] + kernel.shape[0]), color=(255, -1, -1))

        cropped_image = image[
            maxLoc[1]:maxLoc[1] +
            kernel.shape[0],
            maxLoc[0]:maxLoc[0] +
            kernel.shape[1]]
        if self.SHOW_IMAGES == 'all':
            show(image)
            show(cropped_image)
        return cropped_image

    """ get_colony_size measures colonies in an image. It uses the # rows and # cols to chop up an image and measure the cell size. It then outputs a picture of the entire plate with circles around the colonies and an array of the size of the colonies. """

    def get_colony_size(self, image):
        count_missing_measurements = 0
        colony_size = np.zeros((self.NUM_ROWS, self.NUM_COLS))
        colpic = None
        for x in xrange(0, self.NUM_ROWS):
            rowpic = None
            for y in xrange(0, self.NUM_COLS):
                cell = get_sub_image(x, y, image, self.NUM_ROWS, self.NUM_COLS)
                points, pic = detectcolonies(cell)
                assert len(points) < 2
                if len(points) < 1:
                    count_missing_measurements += 1
                    colony_size[x, y] = np.nan
                else:
                    colony_size[x, y] = points[0].size
                if rowpic is None:
                    rowpic = pic
                else:
                    rowpic = np.hstack((rowpic, pic))
            if colpic is None:
                colpic = rowpic
            else:
                colpic = np.vstack((colpic, rowpic))
        # print "less than 1 points: " + str(count_missing_measurements)
        return colony_size, colpic, count_missing_measurements

    def save_to_file(size, filename):
        with open(filename, "wb") as f:
            writer = csv.writer(f)
            writer.writerow(['row', 'column', 'size'])
            for y in xrange(self.NUM_COLS):
                for x in xrange(self.NUM_ROWS):
                    writer.writerow([y, x, size[x, y]])

    """ normalize_edges takes an array of sizes and normalizes the values around the edge of a plate such that the median size of edge colonies is the same as the median size of the colonies in the rest of the plate """

    def normalize_edges(self, sizes):  # TODO: make sure this works when values are missing
        outside_vals = list()
        inside_vals = list()
        for y in xrange(self.NUM_COLS):
            for x in xrange(self.NUM_ROWS):
                if y == 1 or y == self.NUM_COLS - 1 or x == 1 or x == self.NUM_ROWS - 1:
                    outside_vals.append(sizes[x, y])
                else:
                    inside_vals.append(sizes[x, y])
        correction = np.median(inside_vals) / np.median(outside_vals)
        new_sizes = np.empty_like(sizes)
        for y in xrange(self.NUM_COLS):
            for x in xrange(self.NUM_ROWS):
                if y == 1 or y == self.NUM_COLS - 1 or x == 1 or x == self.NUM_ROWS - 1:
                    new_sizes[x, y] = sizes[x, y] * correction
                else:
                    new_sizes[x, y] = sizes[x, y]
        return new_sizes

    def process_files(self, files):
        index = pd.MultiIndex.from_product(
            [range(self.NUM_ROWS), range(self.NUM_COLS)], names=["Row", "Column"])
        colonies_sizes = pd.DataFrame(index=index)
        if not isinstance(files, list):
            files = [files]  # make a list of files if it isn't one already
        for file in files:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            image_cropped = self.kernel_crop(image.copy())
            colony_sizes, image_w_circles, missing_measurements = self.get_colony_size(
                image_cropped)
            n_colony_sizes = self.normalize_edges(colony_sizes)
            colonies_sizes[os.path.splitext(os.path.basename(file))[
                0]] = n_colony_sizes.reshape(-1, 1)

            if missing_measurements > 0:  # Report when missing measurements
                print "Missing {0} for file {1}".format(missing_measurements, file)
                if self.SHOW_IMAGES == 'all' or self.SHOW_IMAGES == 'missing':
                    show(image_w_circles)

            cv2.imwrite(
                './results_images/output_' +
                os.path.basename(file) +
                '.png',
                image_w_circles)

        colonies_sizes.to_csv(path_or_buf='./results_numbers/results.csv')
        return colonies_sizes


def show(image):
    cv2.imshow("Image", imutils.resize(image, width=800))
    cv2.waitKey(0)


def rough_crop(gray_full):
    resized = imutils.resize(gray_full, width=600)
    ratio = gray_full.shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = resized
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
# shape detector
    image, contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# compute the center of the contour, then detect the name of the
# shape using only the contour
# multiply the contour (x, y)-coordinates by the resize ratio,
# then draw the contours and the name of the shape on the image
    largest_contour = None
    largest_contour_area = 0
    for contour in contours:
        contour = np.array(np.array(contour) * ratio, dtype=int)
        area = cv2.contourArea(contour)
        if(area > largest_contour_area):
            largest_contour = contour
            largest_contour_area = area

    #cv2.drawContours(image=gray_full, contours=[largest_contour], contourIdx=-1,thickness = cv2.FILLED, color=(255,0,0))
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_gray = gray_full[y:y + h, x:x + w]
    return cropped_gray


def crop_to_kernel():
    pass


def detectcolonies(image):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 30
# Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 100
# Filter by Area.
    params.filterByArea = True
    params.minArea = 55
    params.maxArea = 1500
# Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
# Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.07
# Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.0
    detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
    keypoints = detector.detect(255 - image)
    largest = keypoints
    if len(keypoints) > 1:
        size = 0
        for keypoint in keypoints:
            if keypoint.size > size:
                largest = [keypoint]
                size = keypoint.size

    im_with_keypoints = cv2.drawKeypoints(image, largest, np.array(
        []), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return largest, im_with_keypoints


def get_sub_image(x, y, image, x_size, y_size):
    stepx = image.shape[0] / x_size
    stepy = image.shape[1] / y_size
    return image[x * stepx:(x + 1) * (stepx), y * stepy:(y + 1) * (stepy)]
    return image[y * stepy:(y + 1) * (stepy), x * stepx:(x + 1) * (stepx)]


def check_is_image(fname):
    return os.path.isfile(fname) and imghdr.what(fname) != None
if __name__ == "__main__":
    db = DBColonySizer()
    file_list = []
    for root, dirs, files in os.walk('/Users/delbalso/Downloads/Gmail'):
        print files
        file_list = [os.path.join(root, file) for file in files]
    print file_list
    db.process_files(file_list)

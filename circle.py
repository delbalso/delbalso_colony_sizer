# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import csv

NUM_COLS = 24
NUM_ROWS = 16

def get_file_list():
# construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image", default = None)
    ap.add_argument("-d", "--dir", required = False, help = "Path to the dir of images", default = None)
    args = vars(ap.parse_args())
    if args["dir"] != None:
        pass
        #files = glob.glob(args["dir"+'/*.{jpg,png,PNG,jpeg}')
    else:
        files = [args["image"]]
    print files
    return files

def kernel_crop(image):
    kernel = cv2.imread("./kernel.PNG", cv2.IMREAD_GRAYSCALE)
    filtered = cv2.matchTemplate(image=image,templ=kernel,method=cv2.TM_CCORR_NORMED)
    minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(filtered)
    cv2.rectangle(img=image,pt1=(maxLoc[0],maxLoc[1]),pt2=(maxLoc[0]+kernel.shape[1],maxLoc[1]+kernel.shape[0]), color=(255,-1,-1))
    show(image)

    cropped_image = image[maxLoc[1]:maxLoc[1]+kernel.shape[0],maxLoc[0]:maxLoc[0]+kernel.shape[1]]
    show(cropped_image)
    return cropped_image

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
    image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
# compute the center of the contour, then detect the name of the
# shape using only the contour
# multiply the contour (x, y)-coordinates by the resize ratio,
# then draw the contours and the name of the shape on the image
    largest_contour = None
    largest_contour_area = 0
    for contour in contours:
        contour = np.array(np.array(contour)*ratio,dtype=int)
        area = cv2.contourArea(contour)
        if(area>largest_contour_area):
            largest_contour = contour
            largest_contour_area = area

    #cv2.drawContours(image=gray_full, contours=[largest_contour], contourIdx=-1,thickness = cv2.FILLED, color=(255,0,0))
    x,y,w,h =cv2.boundingRect(largest_contour)
    cropped_gray = gray_full[y:y+h,x:x+w]
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
    if len(keypoints)>1:
        size = 0
        for keypoint in keypoints:
            if keypoint.size>size :
                largest =[keypoint]
                size = keypoint.size

    im_with_keypoints = cv2.drawKeypoints(image, largest, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return largest, im_with_keypoints

def get_sub_image(x,y,image,x_size,y_size):
    stepx = image.shape[1]/x_size
    stepy = image.shape[0]/y_size
    return image[y*stepy:(y+1)*(stepy),x*stepx:(x+1)*(stepx)]

def show(image):
    cv2.imshow("Image", imutils.resize(image, width=800))
    cv2.waitKey(0)
def get_colony_size(image):
    l1 = 0
    colony_size = np.zeros((NUM_COLS, NUM_ROWS))
    rowpic = None
    for x in xrange(0,NUM_COLS):
        colpic=None
        for y in xrange(0,NUM_ROWS):
            cell = get_sub_image(x,y,image,NUM_COLS,NUM_ROWS)
            points, pic = detectcolonies(cell)
            assert len(points)<2
            if len(points)<1:
                l1+=1
                colony_size[x,y] = np.nan
            else:
                colony_size[x,y] = points[0].size
            if colpic is None:
                colpic = pic
            else:
                colpic = np.vstack((colpic,pic))
        if rowpic is None:
            rowpic = colpic
        else:
            rowpic = np.hstack((rowpic,colpic))
    print "less than 1 points: " + str(l1)
    return colony_size, rowpic

def save_to_file(size, filename):
    with open("output.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'column', 'size'])
        for y in xrange(NUM_ROWS):
            for x in xrange(NUM_COLS):
                writer.writerow([y, x, size[x,y]])

def normalize_edges(sizes):
    outside_vals = list()
    inside_vals = list()
    for y in xrange(NUM_ROWS):
        for x in xrange(NUM_COLS):
            if y == 1 or y == NUM_ROWS-1 or x == 1 or x == NUM_COLS-1:
                outside_vals.append(sizes[x,y])
            else:
                inside_vals.append(sizes[x,y])
    correction = np.median(inside_vals)/np.median(outside_vals)
    new_sizes = np.empty_like(sizes)
    for y in xrange(NUM_ROWS):
        for x in xrange(NUM_COLS):
            if y == 1 or y == NUM_ROWS-1 or x == 1 or x == NUM_COLS-1:
                new_sizes[x,y] = sizes[x,y]*correction
            else:
                new_sizes[x,y] = sizes[x,y]
    return new_sizes

files = get_file_list()
file = files[0]
image  = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
cropped_im = kernel_crop(image.copy())
sizes, image_w_circles = get_colony_size(cropped_im)
sizes = normalize_edges(sizes)
save_to_file(sizes, './output.csv')
cv2.imwrite('./detected.png',image_w_circles)
show(image_w_circles)
print sizes


#!/bin/env python

import os
import cv2 as cv
import numpy as np

margin = 4


def check(w, h):
    return (w > 10 and h > 10) and (w < 1.25 * h) and (h < 1.25 * w)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(
        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    img = cv.imread(img_for_box_extraction_path, 0)

    ret, orig = cv.threshold(img, 140, 255, cv.THRESH_BINARY)

    img = cv.blur(img, (5, 5))

    img_bin = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 11, 0)
    img_bin = 255-img_bin

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv.dilate(img_temp1, verticle_kernel, iterations=3)
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv.dilate(img_temp2, hori_kernel, iterations=3)
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv.threshold(
        img_final_bin, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv.findContours(
        img_final_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv.boundingRect(c)
        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if check(w, h):
            idx += 1
            new_img = orig[y + margin:y+h - margin, x + margin:x+w - margin]
            resized = cv.resize(new_img, (32, 32), interpolation=cv.INTER_AREA)
            ret, resized = cv.threshold(resized, 180, 255, cv.THRESH_BINARY)
            cv.imwrite(cropped_dir_path+str(idx) + '.png', resized)


dirs = [x for x in os.listdir() if os.path.isdir(x)]
print(dirs)

for directory in dirs:
    cnt = 1
    files = ['/'.join([directory, x]) for x in os.listdir(directory)
             if os.path.isfile('/'.join([directory, x]))]
    for file in files:
        print(file)
        out_dir = os.path.dirname(file) + '/crop/' + str(cnt) + '/'
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass
        box_extraction(file, out_dir)
        cnt += 1
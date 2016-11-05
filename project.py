#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pprint
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# %matplotlib inline

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_average_line(img,lines, slopes):
    if lines.size == 0:
        return [0,0,0,0]
    # get line lengths
    # these will be used to weigh the contributions
    # of individual lines to the average
    squaredLineLengths = \
        np.square(lines[:,2]-lines[:,0]) + np.square(lines[:,2]-lines[:,1])
    averageSlope = np.average(slopes, weights=squaredLineLengths)
    # b = y - mx
    intercepts = lines[:,1] - slopes[:] * lines[:,0]
    averageIntercept = np.average(intercepts, weights=squaredLineLengths)

    y = img.shape[0]
    y1 = 0.6*y
    y2 = y
    x1 = (y1 - averageIntercept) / averageSlope
    x2 = (y2 - averageIntercept) / averageSlope
    return [x1, y1, x2, y2]


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    # get all slopes
    # m = (y2 - y1) / (x2-x1)
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    slopes = (lines[:,3] - lines[:,1]) / (lines[:,2] - lines[:,0])
    # check for nan and inf values (horizontal or vertical lines)
    lines = lines[~np.isnan(slopes) & ~np.isinf(slopes)]
    slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]

    # assign lines&slopes to right or left lane marker
    rightLines = lines[(slopes > .5) & (slopes < .9)]
    rightSlopes = slopes[(slopes > .5) & (slopes < .9)]
    leftLines = lines[(slopes < -.5) & (slopes > -.9)]
    leftSlopes = slopes[(slopes < -.5) & (slopes > -.9)]

    l = get_average_line(img, leftLines, leftSlopes)
    r = get_average_line(img, rightLines, rightSlopes)

    cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), color, thickness*3)
    cv2.line(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, thickness*3)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(file):
    gray = grayscale(file)
    blur_gray = gaussian_blur(gray, 5)
    edges = canny(blur_gray, 50, 150)

    imshape = gray.shape
    x = imshape[1]
    y = imshape[0]
    vertices = np.array([[(.05*x,y),(.47*x, .6*y), (.53*x, .6*y), (.95*x,y)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    threshold = 20
    min_line_len = 35
    max_line_gap = 20
    hough_lines_output = hough_lines(masked_edges, 2, np.pi/180, threshold, min_line_len, max_line_gap)

    weighted_image = weighted_img(hough_lines_output, file)
    return weighted_image

# testImages = os.listdir("test_images/")
# for testImage in testImages:
#     image = mpimg.imread('test_images/' + testImage)
#     processedImage = process_image(image)
#     plt.figure()
#     plt.imshow(processedImage)
# plt.show()

version = 0

if version == 0:
    image = mpimg.imread('test_images/solidWhiteRight.jpg')
    processedImage = process_image(image)
    plt.figure()
    plt.imshow(processedImage)
    plt.show()
elif version == 1:
    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
elif version == 2:
    yellow_output = 'yellow.mp4'
    clip2 = VideoFileClip('solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)
elif version == 3:
    challenge_output = 'extra.mp4'
    clip2 = VideoFileClip('challenge.mp4')
    challenge_clip = clip2.fl_image(process_image)
    challenge_clip.write_videofile(challenge_output, audio=False)

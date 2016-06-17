#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy 
from cv2 import cv
# if len(sys.argv) != 3:
    # print(
        # "Give the path to the trained shape predictor model as the first "
        # "argument and then the directory containing the facial images.\n"
        # "For example, if you are in the python_examples folder then "
        # "execute this program by running:\n"
        # "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        # "You can download a trained facial shape predictor from:\n"
        # "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    # exit()

# predictor_path = sys.argv[1]
# faces_folder_path = sys.argv[2]
faceOnly = False
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11
JAW_LINE = list(range(0,17))
# FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
OUTER_LIP = list(range(48,60))
INNER_LIP = list(range(60,68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_BRIDGE = list(range(27,31))
LOWER_NOSE = list(range(30,36))
NOSE_POINTS = list(range(27, 35))

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

OVERLAY_GROUPS = []
OVERLAY_GROUPS = [JAW_LINE,LEFT_BROW_POINTS,RIGHT_BROW_POINTS,JAW_LINE,LEFT_BROW_POINTS,RIGHT_BROW_POINTS,
    NOSE_BRIDGE,LOWER_NOSE,RIGHT_EYE_POINTS,LEFT_EYE_POINTS,OUTER_LIP,INNER_LIP]

# OVERLAY_LINES = [
    # JAW_LINE + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    # NOSE_BRIDGE + LOWER_NOSE + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + OUTER_LIP + INNER_LIP,
        
# ]
screenheight = 320
screenwidth = 480
predictor_path='../../examples/faces/shape_predictor_68_face_landmarks.dat' 
# faces_folder_path = '../examples/faces/'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
x_offset=y_offset=50
emoji = cv2.imread("happy.png")
faceRect = detector(emoji, 1)[0]
# win = dlib.image_window()
def get_landmarks(im):
    global faceRect
    rects = detector(im, 1)
    if rects: faceRect = rects[0]
    landmarks = []
    for idx,i in enumerate(rects):
        landmarks.append(numpy.matrix([[p.x,p.y] for p in predictor(im, rects[idx]).parts()]))
    return landmarks

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for landmark in landmarks:
        for idx, point in enumerate(landmark):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
    
def read_im_and_landmarks():
    ret,im = cam.read()
    # gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         # im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
    
def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    
    for landmark in landmarks:
        for group in OVERLAY_POINTS:
            draw_convex_hull(im,
                             landmark[group],
                             color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def draw_polyline(im,landmarks):
    if faceOnly:
        im = numpy.zeros(im.shape[:3], dtype=numpy.float64)

    pts = numpy.array([[10,5],[20,30],[70,20],[50,10]], numpy.int32)
    pts = pts.reshape((-1,1,2))
    for landmark in landmarks:
        for group in OVERLAY_GROUPS:
            ftrpoints = [landmark[group]]
            cv2.polylines(im,ftrpoints,False,(0,255,0),1,8)
    return im

cam = cv2.VideoCapture(0)
print "Camera initialized"
cam.set(3,screenwidth)
cam.set(4,screenheight)
scale = 0.5
cv2.namedWindow('CamFace')
tickCount = 0
looping = True

while looping:
    tickCount += 1
    x_offset,y_offset = faceRect.left(),faceRect.top()
    if x_offset < 0: x_offset = 0
    if x_offset > screenwidth: x_offset = screenwidth
    if y_offset < 0: y_offset = 0
    if y_offset > screenheight: y_offset = screenheight
    im, s = read_im_and_landmarks()

    # im = annotate_landmarks(im,s)
    # im = get_face_mask(im,s) 
    im = draw_polyline(im,s)
    # cv2.addWeighted(emoji,0.5,im,0.5,0,im)
    im[y_offset:y_offset+emoji.shape[0], x_offset:x_offset+emoji.shape[1]] = emoji
    cv2.imshow('CamFace',im) 
    # win.add_overlay(dets)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress != 255:
        print (keypress)
        if keypress == 32: # Spacebar
            faceOnly = not faceOnly
        elif keypress == 113 or 27: # 'q' pressed to quit
            print "Escape key entered"
            looping = False
  
# When everything is done, release the capture
cam.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

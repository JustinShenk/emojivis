#!/usr/env/bin python3
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
import numpy as np
# import video
# from cv2 import cv
from common import anorm2, draw_str
from time import clock
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

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

# LK parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

track_len = 10
detect_interval = 5
tracks = []
frame_idx = 0

# Other parameters
saveToVideo = False
faceOnly = False
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
JAW_LINE = list(range(0, 17))
# FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
OUTER_LIP = list(range(48, 60))
INNER_LIP = list(range(60, 68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_BRIDGE = list(range(27, 31))
LOWER_NOSE = list(range(30, 36))
NOSE_POINTS = list(range(27, 35))

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

OVERLAY_GROUPS = []
OVERLAY_GROUPS = [JAW_LINE, LEFT_BROW_POINTS, RIGHT_BROW_POINTS, JAW_LINE, LEFT_BROW_POINTS, RIGHT_BROW_POINTS,
                  NOSE_BRIDGE, LOWER_NOSE, RIGHT_EYE_POINTS, LEFT_EYE_POINTS, OUTER_LIP, INNER_LIP]

# OVERLAY_LINES = [
# JAW_LINE + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
# NOSE_BRIDGE + LOWER_NOSE + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + OUTER_LIP + INNER_LIP,

# ]
screenheight = 320
screenwidth = 480

# predictor_path='../../examples/faces/shape_predictor_68_face_landmarks.dat'
# faces_folder_path = '../examples/faces/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
x_offset = y_offset = 50
emoji = cv2.imread("happy.png")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vout = cv2.VideoWriter()
success = vout.open('output.mov', fourcc, 15, (480, 320), True)
faceRect = detector(emoji, 1)[0]
# win = dlib.image_window()


def get_landmarks(im):
    global faceRect
    rects = detector(im, 1)
    if rects:
        faceRect = rects[0]
    landmarks = []
    for idx, i in enumerate(rects):
        landmarks.append(np.matrix([[p.x, p.y]
                                    for p in predictor(im, rects[idx]).parts()]))
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
    ret, im = cam.read()
    # gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
    # im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
    print(points)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for landmark in landmarks:
        for group in OVERLAY_POINTS:
            draw_convex_hull(im,
                             landmark[group],
                             color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

# Draw delaunay triangles
def draw_delaunay(im, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = im.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(im, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(im, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(im, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)
 
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def draw_polyline(im, landmarks):
    if faceOnly:
        print("faceOnly on")
        im[0:screenheight] = 0.0
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    if len(landmarks):
        bottomRight = (max(landmarks[0][:, 0]), max(landmarks[0][:, 1]))
        topLeft = (min(landmarks[0][:, 0]), min(landmarks[0][:, 1]))
        rect = (0,0,im.shape[1],im.shape[0])
        # FIXME: Delaunay triangulation.
        # subdiv = cv2.Subdiv2D(rect)
    for landmark in landmarks:
        for group in OVERLAY_GROUPS:
            ftrpoints = [landmark[group]]            
            cv2.polylines(im, ftrpoints, False, (0, 255, 0), 1, 8)
            # FIXME: Delaunay triangulation.
            # for pt in ftrpoints:
            #     subdiv.insert(pt)
    # FIXME: Delaunay triangulation.
    # if len(landmarks):
    #     draw_delaunay( im, subdiv, (255,255,2555))
    return im


def analyze_emotions(im, landmarks):
    for landmark in landmarks:
        # Observe eyebrow height for surprise
        standheight = np.absolute(landmark[27, 1] - landmark[30, 1])
        eyebrowheight = np.absolute(landmark[27, 1] - landmark[19, 1])
        if standheight == 0:
            standheight += 0.01
        eyedist = float(eyebrowheight) / float(standheight)
        mouthheight = np.absolute(landmark[50, 1] - landmark[57, 1])
        if float(mouthheight) / float(standheight) > 30:
            cv2.putText(im, "mouthheight: " + str(mouthheight), (screenwidth - 80, 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255),
                        thickness=2)
        eyedist += mouthheight / 30
        mouthwidth = np.absolute(landmark[48, 0] - landmark[50, 0])
        nosewidth = np.absolute(landmark[31, 0] - landmark[35, 0])
        mouthdist = float(mouthwidth) / nosewidth
        im = score_emotions(im, eyedist, mouthdist)

    return im


def clean_offset(x_offset, y_offset):
    if x_offset < 0:
        x_offset = 0
    if x_offset > screenwidth:
        x_offset = screenwidth
    if y_offset < 0:
        y_offset = 0
    if y_offset > screenheight:
        y_offset = screenheight
    return x_offset, y_offset


def score_emotions(im, eyebrowheight, mouthdist):
    gray = (129, 129, 129)
    red = (0, 0, 255)
    if eyebrowheight > 0.75:
        surscore = eyebrowheight * 10
        surscore = str(int(surscore))
        color = red
    else:
        color = gray
        surscore = ''
    cv2.putText(im, "SURPRISE = " + surscore, (3 * screenwidth / 5, screenheight / 3),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.7,
                    color=color,
                    thickness=1)
    if mouthdist > 0.9:
        hapscore = mouthdist * 100
        hapscore = str(int(hapscore))
        color = red
    else:
        color = gray
        hapscore = ''
    cv2.putText(im, "HAPPINESS = " + hapscore, (3 * screenwidth / 5, screenheight / 3 + 20),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.7,
                    color=color,
                    thickness=1)

    return im
# Initialize camera
cv2.namedWindow('CamFace')
cam = cv2.VideoCapture(0)

print("Camera initialized")
cam.set(3, screenwidth)
cam.set(4, screenheight)
size = (480, 320)
# size = (int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
# int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
scale = 0.5
tickCount = 0
looping = True

# Video loop
while looping:
    tickCount += 1
    x_offset, y_offset = clean_offset(faceRect.left(), faceRect.top())

    # Detect faces and landmarks
    im, s = read_im_and_landmarks()

    # Draw green lines
    im = draw_polyline(im, s)
    # Other display options
    # im = annotate_landmarks(im,s)
    # im = get_face_mask(im,s)
    # im = analyze_emotions(im,s)

    # Add emoji to image
    im[y_offset:y_offset + emoji.shape[0],
        x_offset:x_offset + emoji.shape[1]] = emoji
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    vis = im.copy()

    # Adding LK_track
    if len(tracks) > 0:
        print("Len > 0")
        img0, img1 = prev_gray, im_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(
            img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        tracks = new_tracks
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(im_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        p = cv2.goodFeaturesToTrack(im_gray, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    frame_idx += 1
    prev_gray = im_gray
    # Show image
    # cv2.imshow('CamFace',im)
    cv2.imshow('CamFace', vis)

    # Save to video (optional)
    if saveToVideo:
        vout.write(im)

    keypress = cv2.waitKey(5) & 0xFF
    if keypress != 255:
        print(keypress)
        if keypress == 32:  # Spacebar
            faceOnly = not faceOnly
        elif keypress == 113 or 27:  # 'q' pressed to quit
            print("Escape key entered")
            looping = False

# When everything is done, release the capture
cam.release()
vout.release()
vout = None
cv2.waitKey(0)
cv2.destroyAllWindows()

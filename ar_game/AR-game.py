import cv2
#import cv2.aruco as aruco
from cv2 import aruco
import numpy as np
import pyglet
from PIL import Image
import sys
import random

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# define aruco dictionary and parameters
# aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# aruco_params = aruco.DetectorParameters()

# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

# holds the code for aruco marker detection and warping the image
class ArucoDetector():

    def __init__(self) -> None:
        self.corners = []
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        self.detected = False

    def setFrame(self, frame):
        self.frame = frame

    def getFrame(self):
        return self.frame
    
    def order_points(self, pts):
        # desperate hour long google search brought me here:
        # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # gets the current frame from webcam
    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            #aruco.drawDetectedMarkers(frame, corners)

            # left upper corner gets appended to array
            for c in corners:
                self.corners.append([c[0][0][0], c[0][0][1]])

            # array of left upper corners gets sorted
            # left up marker is now first in array
            ordered_corners = self.order_points(np.array(self.corners))

            # as soon as all markers ar detected, the warping/transformation of image is done
            if len(self.corners) == 4:
                selection_points = np.float32(ordered_corners)
                selection_points[3][0] += 20
                warp_points = np.float32([[0,0], [WINDOW_WIDTH,0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0,WINDOW_HEIGHT]])

                matrix = cv2.getPerspectiveTransform(selection_points, warp_points)
                img_warp = cv2.warpPerspective(frame, matrix, (WINDOW_WIDTH, WINDOW_HEIGHT),flags=cv2.INTER_LINEAR)

                self.frame = img_warp
                self.detected = True

            else: # if no all markers are detected: display normal webcam picture
                self.frame = frame
                self.detected = False
            # pretends overflow with detected corners
            self.corners.clear()
        else: # displays normal webcam picture if no markers at all are being detected
            self.frame = frame 
            self.detected = False

# holds the contouring/threshold code for fingerdetection
class ContourDetector():

    def __init__(self) -> None:
        self.threshold = 140 # worked best in my light situation, may be different in different rooms
        self.out = None
        self.collided = False
        self.collided_black = False

    def set_frame(self, frame):
        self.frame = frame
    
    # code from: https://github.com/madhav727/hand-detection-and-finger-counting/blob/master/finger_counting_video.py
    # because my old threshold code couldn't detect fingers steady
    def skinmask(self, img):
        hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        skinRegionHSV = cv2.inRange(hsvim, lower, upper)
        blurred = cv2.blur(skinRegionHSV, (2,2))
        ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
        return thresh
    
    def is_collided(self):
        return self.collided
    
    def is_black_rec_coll(self):
        return self.collided_black

    def detect_collision(self, frame):
        self.collided_black = False
        self.collided = False
        mask_img = self.skinmask(frame) # gets better detection of finger through colorspace

        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # collision check with contour and game items
        for item in GameItems.items:
            for contour in contours:
                x = contour[0][0][0]
                y = contour[0][0][1]

                y2 = WINDOW_HEIGHT - y # because coordinates work differently in opencv and pyglet

                # collision detection 
                # finger has to be left of finger slide (black rectangle)
                if (x + 10 >= item.x and x <= GameItems.FINGER_SLIDER_X 
                    and y2 <= item.y + 50 + 30 and y2 >= item.y - 30):
                    GameItems.items.remove(item)
                    self.collided = True
                    if item.color == (0,0,0,255):
                        self.collided_black = True
                    break

# holds all game items (rectangles to destroy, labels)
class GameItems:

    items = []
    FINGER_SLIDER_X = 120 # x coordinate of finger slide

    def __init__(self) -> None:
        self._start_y_points = [100, 180, 260, 340]
        self._colors = [(0,0,0), (159, 126, 105), (166, 195, 111), (0, 78, 137)]
        self.width = 50
        self.height = 50
        self.x = WINDOW_WIDTH  + self.width
        self.finger_slider = pyglet.shapes.Rectangle(x=self.FINGER_SLIDER_X, y=0, width=2, height=WINDOW_HEIGHT, color=(20,20,20))
        self.score = 0
        self.score_label = pyglet.text.Label('Score: 0 | Lives: 3', font_name="Times New Roman",
                                             font_size=20,x=WINDOW_WIDTH /2, y=WINDOW_HEIGHT-50, anchor_x='center',
                                             color=(10 ,10,10,255))
        self.end_label = pyglet.text.Label('You died, press Q to quit', font_name="Times New Roman",
                                             font_size=40,x=WINDOW_WIDTH /2, y=WINDOW_HEIGHT/2, anchor_x='center',
                                             anchor_y='center', color=(150 , 150, 150 ,255))
        self.lives = 3

    # creates new rectangles to be destroyed with finger interaction
    def create_item(self):
        if random.randint(0,15) == 0: # change 15 to whatever to fasten up / slow down the game
            y = self._start_y_points[random.randint(0, len(self._start_y_points)-1)]
            color = self._colors[random.randint(0, len(self._colors)-1)]
            rec = pyglet.shapes.Rectangle(x=self.x, y=y, width=self.width, height=self.height, color=color)
            GameItems.items.append(rec)

    def draw_items(self):
        self.finger_slider.draw()
        self.score_label.draw()
        for item in GameItems.items:
            item.draw()

    def update_items(self):
        for item in GameItems.items:
            item.x -= 10

            if item.x + self.width <= 0:
                GameItems.items.remove(item)

    def draw_end(self):
        self.end_label.draw()

    def check_alive(self):
        if self.lives == 0:
            return False
        else:
            return True

    def add_score(self):
        self.score += 15
        self.score_label.text = f"Score: {self.score} | Lives: {self.lives}"
    
    def minus_score(self):
        self.score -= 20
        if self.lives >= 1:
            self.lives -= 1
            self.score_label.text = f"Score: {self.score} | Lives: {self.lives}"


cap = cv2.VideoCapture(video_id)

WINDOW_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# WINDOW_WIDTH = 640
# WINDOW_HEIGHT = 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
arucoDet = ArucoDetector()
contourDet = ContourDetector()
game = GameItems()

@window.event
def on_show():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0) # remove this when CAP_PROB_FRAME removed
    arucoDet.setFrame(frame)
    contourDet.set_frame(frame) 

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        sys.exit(0)


@window.event
def on_draw():
    window.clear()
 
    ret, frame = cap.read()

    arucoDet.detect_markers(frame)
    test = arucoDet.getFrame()

    if arucoDet.detected:
        if game.check_alive():
            game.create_item()
            game.update_items()
            contourDet.detect_collision(test)

            if (contourDet.is_collided() and contourDet.is_black_rec_coll()):
                # rectangle is destroyed but it was the black one
                game.minus_score()
            elif (contourDet.is_collided() and not contourDet.is_black_rec_coll()):
                # rectangle is destroyed and it was a colored one
                game.add_score()

            img = cv2glet(test, 'BGR')
            img.blit(0, 0, 0) 
            game.draw_items()
        else:
            game.draw_end()

    else: # no detection of markers -> show normal webcam
        img = cv2glet(test, 'BGR')
        img.blit(0, 0, 0)
        if not game.check_alive():
            game.draw_end()

if __name__ == '__main__':
    pyglet.app.run()

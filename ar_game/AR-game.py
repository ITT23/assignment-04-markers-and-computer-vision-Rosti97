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
        # desperate hour long google search:
        # https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
        # i feel dumb for not knowing any other solution ups
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            #aruco.drawDetectedMarkers(frame, corners)

            for c in corners:
                self.corners.append([c[0][0][0], c[0][0][1]])
            
            #print(f"Length: {len(self.corners)}")

            orderd_stuff = self.order_points(np.array(self.corners))

            if len(self.corners) == 4:
                # print(".")
                selection_points = np.float32(orderd_stuff)
                warp_points = np.float32([[0,0], [WINDOW_WIDTH,0], [WINDOW_WIDTH, WINDOW_HEIGHT], [0,WINDOW_HEIGHT]])

                # print(selection_points)
                # print(warp_points)
                # print("--")


                matrix = cv2.getPerspectiveTransform(selection_points, warp_points)
                img_warp = cv2.warpPerspective(frame, matrix, (WINDOW_WIDTH, WINDOW_HEIGHT),flags=cv2.INTER_LINEAR)

                self.frame = img_warp
                self.detected = True

            else:
                self.frame = frame
                self.detected = False
            
            self.corners.clear()

        else:
            self.frame = frame 


class ContourDetector():

    def __init__(self) -> None:
        self.threshold = 140
        self.out = None
        pass

    def set_frame(self, frame):
        self.frame = frame
    
    # https://github.com/madhav727/hand-detection-and-finger-counting/blob/master/finger_counting_video.py
    def skinmask(self, img):
        hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        skinRegionHSV = cv2.inRange(hsvim, lower, upper)
        blurred = cv2.blur(skinRegionHSV, (2,2))
        ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY)
        return thresh

    def detect_collision(self, frame):
        mask_img = self.skinmask(frame)

        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = max(contours, key=lambda x: cv2.contourArea(x))
        #hull = cv2.convexHull(contours)

        c = cv2.drawContours(frame, contours, -1, (255,255,0), 2)

        for item in GameItems.items:
            for contour in contours:
                x = contour[0][0][0]
                y = contour[0][0][1]

                # x != 0 and and y <= WINDOW_WIDTH-item.y
                if x >= item.x and x <= item.x+50 and x <= GameItems.FINGER_SLIDER_X + 50:
                    GameItems.items.remove(item)
                    break
                #t.color = (0,0,255)
                #print(f"{x},{y}")
                #break

                #t.color = (0,255,0)
        self.out = c

class GameItems:

    items = []
    FINGER_SLIDER_X = 120

    def __init__(self) -> None:
        self._start_y_points = [100, 180, 260, 340]
        self._colors = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,0,0), (0,255,0), (0,0,255)]
        self.width = 50
        self.height = 50
        self.x = WINDOW_WIDTH  + self.width
        self.finger_slider = pyglet.shapes.Rectangle(x=self.FINGER_SLIDER_X, y=0, width=2, height=WINDOW_HEIGHT, color=(20,20,20))

    def create_item(self):
        if random.randint(0,40) == 0:
            y = self._start_y_points[random.randint(0, len(self._start_y_points)-1)]
            color = self._colors[random.randint(0, len(self._colors)-1)]
            rec = pyglet.shapes.Rectangle(x=self.x, y=y, width=self.width, height=self.height, color=color)
            GameItems.items.append(rec)

    def draw_items(self):
        self.finger_slider.draw()
        for item in GameItems.items:
            item.draw()

    def update_items(self):
        for item in GameItems.items:
            item.x -= 10

            if item.x + self.width <= 0:
                GameItems.items.remove(item)
    


cap = cv2.VideoCapture(video_id)

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
detector = ArucoDetector()
contourd = ContourDetector()
game = GameItems()
# t = pyglet.shapes.Rectangle(x=100, y= 90, width=50, height=50, color=(50,225,30))
# z = pyglet.shapes.Rectangle(x=100, y= 180, width=50, height=50, color=(50,225,30))
# u = pyglet.shapes.Rectangle(x=100, y= 270, width=50, height=50, color=(50,225,30))
# i = pyglet.shapes.Rectangle(x=100, y= 360, width=50, height=50, color=(50,225,30))


@window.event
def on_show():
    ret, frame = cap.read()
    detector.setFrame(frame)
    contourd.set_frame(frame)

    

@window.event
def on_draw():
    window.clear()
 
    ret, frame = cap.read()
    #img = cv2glet(frame, 'BGR')

    ret, frame = cap.read()

    detector.detect_markers(frame)
    test = detector.getFrame()

    if detector.detected:
        contourd.detect_collision(test)
        img = cv2glet(contourd.out, 'BGR')
        img.blit(0, 0, 0)
        game.create_item()

        game.update_items()

        game.draw_items()
        
    else:
        img = cv2glet(test, 'BGR')
        img.blit(0, 0, 0)

    
    

    # Convert the frame to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    #test = np.array(ids)
    

    # Check if marker is detected
    # if ids is not None:
    #     # Draw lines along the sides of the marker
    #     aruco.drawDetectedMarkers(frame, corners)

    # else:
    #     # Display the frame
    #     #cv2.imshow('frame', frame)
    #     img = cv2glet(frame, 'BGR')

    # Wait for a key press and check if it's the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("ups")

    #img.blit(0, 0, 0)

    # game.create_item()

    # game.update_items()

    # game.draw_items()
    # t.draw()
    # z.draw()
    # u.draw()
    # i.draw()

pyglet.app.run()

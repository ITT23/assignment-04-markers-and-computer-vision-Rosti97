import cv2
#import cv2.aruco as aruco
from cv2 import aruco
import numpy as np
import pyglet
from PIL import Image
import sys

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
            aruco.drawDetectedMarkers(frame, corners)

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


    # Rectangle 
    # Wir gehen für jeden contours in Contours durch
    # ob x/y zwischen x/y von Rectangle liegt


    def __init__(self) -> None:
        self.threshold = 100
        self.out = None
        pass

    def set_frame(self, frame):
        self.frame = frame

    def measure_distance(self, x1:int, y1:int, x2:int, y2:int) -> float:
        """measures the distance between two coordinates"""
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

    def detect_collision(self, frame, t):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, tresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)

        contours, hierachy = cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_contours = cv2.drawContours(frame, contours, -1, (255,0,0), 3)

        test_array = []

        for contour in contours:
            x = contour[0][0][0]
            y = contour[0][0][1]

            if x != 0 and x >= 400 and x <= 600 and y>=410:
              t.color = (0,0,255)
              print(x)



        self.out =  img_contours


cap = cv2.VideoCapture(video_id)

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
detector = ArucoDetector()
contourd = ContourDetector()

@window.event
def on_show():
    ret, frame = cap.read()
    detector.setFrame(frame)
    contourd.set_frame(frame)
    

@window.event
def on_draw():
    window.clear()

    t = pyglet.shapes.Rectangle(x=400, y= 410, width=50, height=50, color=(50,225,30))
    ret, frame = cap.read()
    #img = cv2glet(frame, 'BGR')

    ret, frame = cap.read()

    detector.detect_markers(frame)
    test = detector.getFrame()

    if detector.detected:
        contour_test = contourd.detect_collision(test, t)
        img = cv2glet(contourd.out, 'BGR')
    else:
        img = cv2glet(test, 'BGR')

    
    

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

    img.blit(0, 0, 0)
    t.draw()
pyglet.app.run()

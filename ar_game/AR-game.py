import cv2
#import cv2.aruco as aruco
from cv2 import aruco
import numpy as np
import pyglet
from PIL import Image
import sys
import random

video_id = 0
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

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
            self.detected = False


class ContourDetector():

    def __init__(self) -> None:
        self.threshold = 140
        self.out = None
        self.collided = False
        self.collided_black = False

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
    
    def is_collided(self):
        return self.collided
    
    def is_black_rec_coll(self):
        return self.collided_black

    def detect_collision(self, frame):
        self.collided_black = False
        self.collided = False
        mask_img = self.skinmask(frame)

        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = max(contours, key=lambda x: cv2.contourArea(x))
        #hull = cv2.convexHull(contours)

        c = cv2.drawContours(frame, contours, -1, (255,255,0), 2)

        for item in GameItems.items:
            for contour in contours:
                x = contour[0][0][0]
                y = contour[0][0][1]

                y2 = WINDOW_HEIGHT - y
                #print(f"x: {x}, y: {y2}")

                #print(f"item1: {WINDOW_HEIGHT - GameItems.items[0].y}, i2: {WINDOW_HEIGHT-item.y-50},  y: {y}")
                #print(f"item1: {GameItems.items[0].y}, i2: {item.y-50},  y: {y}")

                # x != 0 and and y <= WINDOW_WIDTH-item.y
                # x <= item.x+50
                if (x + 50 >= item.x and x <= GameItems.FINGER_SLIDER_X + 50
                    #and y <= WINDOW_HEIGHT-item.y-70 and y >= WINDOW_HEIGHT-item.y-50+70 ):
                    and y2 <= item.y + 50 + 30 and y2 >= item.y - 30):
                    GameItems.items.remove(item)
                    self.collided = True
                    if item.color == (0,0,0,255):
                        self.collided_black = True
                    break
                #t.color = (0,0,255)
                #print(f"{x},{y}")
                #break

                #t.color = (0,255,0)
       # self.out = c

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
        self.score = 0
        self.score_label = pyglet.text.Label('Score:', font_name="Times New Roman",
                                             font_size=20,x=WINDOW_WIDTH /2, y=WINDOW_HEIGHT-50, anchor_x='center',
                                             color=(232,98,82,255))

    def create_item(self):
        if random.randint(0,25) == 0:
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

    def add_score(self):
        self.score += 15
        self.score_label.text = f"Score: {self.score}"
    
    def reset_score(self):
        self.score += 15
        self.score_label.text = f"Score: {self.score}"


cap = cv2.VideoCapture(video_id)

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
arucoDet = ArucoDetector()
contourDet = ContourDetector()
game = GameItems()
t = pyglet.shapes.Rectangle(x=0, y=300, width=WINDOW_WIDTH, height=2, color=(20,20,20))
t2 = pyglet.shapes.Rectangle(x=0, y=100, width=WINDOW_WIDTH, height=2, color=(20,20,20))

@window.event
def on_show():
    ret, frame = cap.read()
    arucoDet.setFrame(frame)
    contourDet.set_frame(frame) 

@window.event
def on_draw():
    window.clear()
 
    ret, frame = cap.read()
    #img = cv2glet(frame, 'BGR')

    ret, frame = cap.read()

    arucoDet.detect_markers(frame)
    test = arucoDet.getFrame()

    # t = pyglet.text.Label('Score:', font_name="Times New Roman",
    #                                          font_size=20,x=WINDOW_WIDTH /2, y=WINDOW_HEIGHT-50, anchor_x='center',
    #                                          color=(232,98,82,255))


    if arucoDet.detected:
        game.create_item()
        game.update_items()
        contourDet.detect_collision(test)
        if (contourDet.is_collided() and contourDet.is_black_rec_coll()):
            print("death")
        elif (contourDet.is_collided() and not contourDet.is_black_rec_coll()):
            game.add_score()
        img = cv2glet(test, 'BGR')
        img.blit(0, 0, 0) 
        game.draw_items()
        t.draw()
        t2.draw()
        
    else:
        img = cv2glet(test, 'BGR')
        img.blit(0, 0, 0)

    #t.draw()

    
    # Wait for a key press and check if it's the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("ups")


if __name__ == '__main__':
    pyglet.app.run()

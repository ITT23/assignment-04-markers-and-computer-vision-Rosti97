import cv2
import argparse
import numpy as np
import sys

class ImageExtractor:

    WINDOW_NAME = 'Preview Window'

    def __init__(self, path_in, path_out, width, heigth):
        try:
            self.img = cv2.imread(path_in)
            #print(self.img)
        except:
            self.img = cv2.imread('sample_image.jpg')
            print("path couldn't load, example picture will be shown")
        self.origin_img = self.img.copy()
        self.path_out = path_out
        self.width = width
        self.heigth = heigth
        self.original = self.img.copy()
        self.warped_image = None
        self.selected_corners = []
        self.max_corners = 4

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.img = cv2.circle(self.img, (x,y), 5, (255,0,0), -1)
            self.selected_corners.append((x,y))
            cv2.imshow(self.WINDOW_NAME, self.img)

            if len(self.selected_corners) == 4:
                self.warp_image()
        
    def warp_image(self):
        width = 500
        height = 300

        selection_points = np.float32(self.selected_corners)
        warp_points = np.float32([[0,0], [width, 0], [width, height], [0, height]])

        matrix = cv2.getPerspectiveTransform(selection_points, warp_points)
        self.warped_image = cv2.warpPerspective(self.origin_img, matrix, (width, height))

        cv2.imshow(self.WINDOW_NAME, self.warped_image)

    def show(self):
        while True:
            cv2.imshow(self.WINDOW_NAME, self.img)
            #hextractor.keyboard_listener()

            key = cv2.waitKey(0)
            if key == 27:  # restart on esc
                extractor.selected_corners.clear()
                self.img = self.original.copy()
            elif key == ord('s'):
                cv2.imwrite(self.path_out, self.warped_image)
                print("File saved!")
            elif key == ord("q"):          
                break
        cv2.destroyAllWindows()    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for specifying image extraction')

    parser.add_argument('-in', '--input', type=str, help='path of image file to be extracted')
    parser.add_argument('-out', '--output', type=str, help='path of where extraced image file should be saved')
    parser.add_argument('-width', '--width', type=int, help='wished width of extraced image output')
    parser.add_argument('-heigth', '--heigth', type=int, help='wished height of extraced image output')

    args = parser.parse_args()

    #print(args.input)

    if (args.input is not None and args.output is not None and 
        args.width is not None and args.heigth is not None):
        extractor = ImageExtractor(args.input, args.output, args.width, args.width)
    else:
        print("Please parse all needed Arguments (Input, Output, Width, Heigth)!")
        print("Example: python3 image_extractor.py -in sample_image.jpg -out final.jpg -width 600 -heigth 400")
        sys.exit()

    #extractor = ImageExtractor('image_extraction/sample_image.jpg', 'test.jpg', 400, 600)
    cv2.namedWindow(extractor.WINDOW_NAME)
    cv2.setMouseCallback(extractor.WINDOW_NAME, extractor.mouse_callback)
    extractor.show()

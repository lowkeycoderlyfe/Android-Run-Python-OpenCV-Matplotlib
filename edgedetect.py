from kivy.clock import mainthread
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
import numpy as np
import cv2
from camera4kivy import Preview

class EdgeDetect(Preview):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzed_texture = None

    ####################################
    # Analyze a Frame - NOT on UI Thread
    ####################################

    def analyze_pixels_callback(self, pixels, image_size, image_pos, scale, mirror):
        # pixels : analyze pixels (bytes)
        # image_size   : analyze pixels size (w,h)
        # image_pos    : location of Texture in Preview (due to letterbox)
        # scale  : scale from Analysis resolution to Preview resolution
        # mirror : true if Preview is mirrored
        
        rgba   = np.fromstring(pixels, np.uint8).reshape(image_size[1],
                                                         image_size[0], 4)
        # Note, analyze_resolution changes the result. Because with a smaller
        # resolution the gradients are higher and more edges are detected.
        
        # ref https://likegeeks.com/python-image-processing/
        # gray   = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        # blur   = cv2.GaussianBlur(gray, (3,3), 0)
        # edges  = cv2.Canny(blur,50,100)
        # rgba   = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGBA)

        vid(frame=rgba)
        cv2.setMouseCallback('light sources', process_click)
        pixels = rgba.tostring()
        self.make_thread_safe(pixels, image_size) 

    @mainthread
    def make_thread_safe(self, pixels, size):
        if not self.analyzed_texture or\
           self.analyzed_texture.size[0] != size[0] or\
           self.analyzed_texture.size[1] != size[1]:
            self.analyzed_texture = Texture.create(size=size, colorfmt='rgba')
            self.analyzed_texture.flip_vertical()
        self.analyzed_texture.blit_buffer(pixels, colorfmt='rgba')

    ################################
    # Annotate Screen - on UI Thread
    ################################

    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        # texture : preview Texture
        # size    : preview Texture size (w,h)
        # pos     : location of Texture in Preview Widget (letterbox)
        # Add the analyzed image
        if self.analyzed_texture:
            Color(1,1,1,1)
            Rectangle(texture= self.analyzed_texture,
                      size = tex_size, pos = tex_pos)




selections = {}
selection_number = 0

camera_num = 1

working_devices = {}


def found_selected(fx, fy, fr):
    for s in selections:
        x = selections[s][0]
        y = selections[s][1]
        r = selections[s][2]
        if (fx - x) * (fx - x) + (fy - y) * (fy - y) < fr * fr:
            if fr > r:
                selections[s] = (fx, fy, fr)
            return True
def find_light_sources(image, thresh):
    med = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (15, 15), 0)
    mean, stddev = cv2.meanStdDev(med)
    light_bottom_threshold = mean[0][0] + 1 * stddev[0][0]
    if light_bottom_threshold < thresh:
        light_bottom_threshold = thresh
    elif light_bottom_threshold > 255:
        light_bottom_threshold = 255

    mat = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (15, 15), 0)
    ret, mat = cv2.threshold(mat, light_bottom_threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mat, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        print("FOOOO COUNTOUR")
        x, y, w, h = cv2.boundingRect(c)
        center, radius = cv2.minEnclosingCircle(c)
        center_point = (int(center[0]), int(center[1]))
        area = 3.14 * radius * radius
        if 100 <= area <= 50000:
            print("FOOOO area")
            if found_selected(int(center[0]), int(center[1]), radius):
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            r = ((x, y), (x + w, y + h))
            image = cv2.circle(image, center_point, int(radius), color)
    for s in selections:
        cv2.putText(image, s, (selections[s][0], selections[s][1]), cv2.QT_FONT_NORMAL, 0.5, (0, 0, 255))
def process_click(event, x, y, flags, param, name=None):
    global selections
    global selection_number

    if event == cv2.EVENT_LBUTTONDOWN:
        name = name if name is not None else chr(ord('A') + selection_number)
        print(name, x, y)
        selections[name] = (x, y, 0)
        selection_number += 1
def vid(frame):
    thresh = 230
    intensities = {}
    frames_rem = 0
    # ax_fooo = axs[2]
    freqs = np.arange(0, 7 + 1, 1)
    # print(freqs)
    spectrum = np.absolute(np.fft.rfft([0, 2, 5, 30, 40, 0, 2, 5]))
    # print(spectrum)
    find_light_sources(image=frame, thresh=230)
    # cap = cv2.VideoCapture(camera_num)
    # if not cap.isOpened():
    #     print("Cannot open camera")
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # cap_frames = 2 * fps
    # ret, frame = cap.read()










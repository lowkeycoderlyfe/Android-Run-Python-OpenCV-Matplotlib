from kivy.clock import Clock, mainthread
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
import numpy as np
import cv2
import threading
cv = cv2
from camera4kivy import Preview
from gestures4kivy import CommonGestures

# import matplotlib
# matplotlib.use('module://foo.backend_kivy')
from kivy_garden_local_copy.matplotlib.backend_kivyagg import FigureCanvasKivy
import matplotlib.pyplot as plt
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App

class EdgeDetect(Preview, CommonGestures):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzed_texture = None
        Clock.schedule_once(self.after_init)

    def after_init(self, dt):
        global ax_td
        global ax_fd
        global canvas
        fig, axs = plt.subplots()
        temp_fig, temp_axs = plt.subplots()
        # ax_td = axs[0]
        # ax_fd = axs[1]
        ax_td = temp_axs
        ax_fd = axs
        ax_td.set_ylabel("Intensity over Time")
        ax_td.set_xlabel("Frame (@%dfps)" % fps)
        ax_fd.set_ylabel("Intensity by Frequency", size=24)
        # ax_fd.set_xlabel("F (Hz)")
        ax_fd.set_xlabel("F (f_scalar)", size=24)
        canvas = FigureCanvasKivy(fig)
        root = App.get_running_app().root
        # Undo what Preview constructor does
        root.remove_widget(self)
        box = BoxLayout(orientation='vertical', spacing=20)
        box.add_widget(self)
        box.add_widget(canvas)
        root.add_widget(box)

    def cg_tap(self, touch, x, y):
        w, h = 405, 720
        print("Touch:", x, y, touch.sx, touch.sy, touch.ox, touch.oy)
        x = int(touch.sx*w)
        y = (h - int(touch.sy*h))
        if y < 0:
            print('NEGATIVE!', y)
            y = 0
        process_click(x, y)

    def cg_long_press(self, touch, x, y):
        selections_lock.acquire()
        global selections
        global selection_number
        selections = {}
        selection_number = 0
        selections_lock.release()
    ####################################
    # Analyze a Frame - NOT on UI Thread
    ####################################

    def analyze_pixels_callback(self, pixels, image_size, image_pos, scale, mirror):
        # pixels : analyze pixels (bytes)
        # image_size   : analyze pixels size (w,h)
        # image_pos    : location of Texture in Preview (due to letterbox)
        # scale  : scale from Analysis resolution to Preview resolution
        # mirror : true if Preview is mirrored
        # print("Pixels:", image_size, scale)
        rgba   = np.fromstring(pixels, np.uint8).reshape(image_size[1],
                                                         image_size[0], 4)
        # Note, analyze_resolution changes the result. Because with a smaller
        # resolution the gradients are higher and more edges are detected.

        # ref https://likegeeks.com/python-image-processing/
        # gray   = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        # blur   = cv2.GaussianBlur(gray, (3,3), 0)
        # edges  = cv2.Canny(blur,50,100)
        # rgba   = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGBA)
        selections_lock.acquire()
        vid(frame=rgba)
        selections_lock.release()
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
            # print("UI texture: ", tex_size)
            Color(1,1,1,1)
            Rectangle(texture= self.analyzed_texture,
                      size = tex_size, pos = tex_pos)

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
        x, y, w, h = cv2.boundingRect(c)
        center, radius = cv2.minEnclosingCircle(c)
        center_point = (int(center[0]), int(center[1]))
        area = 3.14 * radius * radius
        if 100 <= area <= 50000:
            if found_selected(int(center[0]), int(center[1]), radius):
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            r = ((x, y), (x + w, y + h))
            image = cv2.circle(image, center_point, int(radius), color)
    for s in selections:
        cv2.putText(image, s, (selections[s][0], selections[s][1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.FILLED)


def process_click(x, y, name=None):
    selections_lock.acquire()
    global selections
    global selection_number
    name = name if name is not None else chr(ord('A') + selection_number)
    print(name, x, y)
    selections[name] = (x, y, 0)
    selection_number += 1
    selections_lock.release()


selections_lock = threading.Lock()
selections = {}
selection_number = 0
ax_td = None
ax_fd = None
canvas = None
thresh = 230
intensities = {}
frames_rem = 0
freqs = np.arange(0, 7 + 1, 1)
fps = 30
cap_frames = 2 * fps


def vid(frame):
    global thresh
    global intensities
    global frames_rem
    global freqs
    global fps
    global cap_frames
    global ax_td
    global ax_fd
    global canvas
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    find_light_sources(frame, thresh)
    # Display the resulting frame
    gray = cv.rectangle(gray, (294, 105), (312, 132), (255, 0, 0))
    if frames_rem:
        for s in selections:
            if s in intensities:
                # pixel = gray[selections['A'][1], selections['A'][0]]
                circle_mask = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)
                circle = selections[s]
                radius = circle[2]
                cv.circle(circle_mask, (circle[0], circle[1]), int(radius), (255, 255, 255), 5)
                mean = cv.mean(gray, circle_mask)
                intensities[s].append(mean[0])
        frames_rem -= 1
        if not frames_rem:
            ax_td.cla()
            ax_fd.containers = []
            ax_fd.collections = []
            ax_fd.lines = []
            # ax_fd.cla()
            ax_fd.grid(True)
            for s in selections:
                if s in intensities and len(intensities[s]) == cap_frames:
                    spectrum = np.absolute(np.fft.rfft(intensities[s]))
                    spectrum[0] = 0
                    y_axis_data = spectrum[:len(freqs)]
                    td_lines = ax_td.plot(intensities[s], label=s)
                    markerline, stemlines, _ = ax_fd.stem(freqs, y_axis_data, label=s, basefmt="None",
                                                          use_line_collection=True)
                    plt.setp(markerline, 'color', plt.getp(td_lines[0], 'color'))
                    plt.setp(stemlines, 'color', plt.getp(td_lines[0], 'color'))
                    # Plot derivative. A->A'
                    # ax_fooo.plot(freqs, np.gradient(y_axis_data), label=s+"'", marker='o')
            ax_fd.legend(loc='best')
            draw_canvas_thread_safe()
            for s in selections:
                intensities[s] = []
            frames_rem = cap_frames
    elif len(selections):
        for s in selections:
            intensities[s] = []
        frames_rem = cap_frames

@mainthread
def draw_canvas_thread_safe():
    global canvas
    canvas.draw()

# This is a sample Python script.
import cv2
import time
import sys
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame  # pygame lib is used to display frames using pyopenGL [pip install pygame]
import numpy as np
# from pygrabber.dshow_graph import FilterGraph
from multiprocessing import Process, Queue, Event, freeze_support, active_children, Value
from multiprocessing.shared_memory import SharedMemory
# pyopenGL uses GPU to render frames [pip install pyopenGL]
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def run_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera failed to open.")
        exit(0)
    print(f"Camera at: {camera}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    ret, frame = camera.read()
    if ret is False:
        print("Camera failed to capture a frame.")
        exit(0)
    print(f"Frame shape: {frame.shape}")
    fps = 0
    pT = time.perf_counter()

    image_4k = np.random.randint(0, 255, (2160, 3840, 3), np.uint8)
    # pygame.init()
    fh, fw, fc = frame.shape  # determine and store height, width and channels after ignoring top lines
    print(f"\nDisplay Window: [{fw} x {fh}]")
    lbl = f"Camera {fh} x {fw}"

    # Initialize the display window
    display = (fw, fh)  # Initialize display size in tuple
    # pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
    # pygame.display.set_caption("Camera")  # Set window title
    # # Set up the OpenGL viewport and projection matrix
    # glViewport(0, 0, display[0], display[1])
    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # gluOrtho2D(0.0, display[0], 0.0, display[1])
    #
    # # Set up the OpenGL modelview matrix
    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()
    #
    # # Prepare a texture for the image (frame)
    # texture = glGenTextures(1)
    # glBindTexture(GL_TEXTURE_2D, texture)
    # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    # print("\nDisplay Renderer (GPU): [", glGetString(GL_RENDERER).decode("utf-8"), "]\n")
    while True:
        fps += 1
        cT = time.perf_counter()
        if cT - pT >= 1:
            print(f"FPS: {fps}")
            fps = 0
            pT = cT

        # ret, frame = camera.read()
        camera.read(frame)
        # Covert image (frame) to texture;
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, display[0], display[1], 0, GL_RGB, GL_UNSIGNED_BYTE, frame)
        #
        # # Draw the image to the screen
        # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # clear screen
        # glEnable(GL_TEXTURE_2D)
        # glBindTexture(GL_TEXTURE_2D, texture)
        # glBegin(GL_QUADS)
        # glTexCoord2f(0, 0)
        # glVertex2f(0, 0)
        # glTexCoord2f(1, 0)
        # glVertex2f(display[0], 0)
        # glTexCoord2f(1, 1)
        # glVertex2f(display[0], display[1])
        # glTexCoord2f(0, 1)
        # glVertex2f(0, display[1])
        # glEnd()
        # pygame.display.flip()  # Display frames
        #
        # # Handle PyGame events to close application
        # for event in pygame.event.get():
        #     if event.type == KEYDOWN:  # event from keyboard
        #         if event.key == K_ESCAPE:  # check whether 'esc' is pressed?
        #             running = False
        #     if event.type == QUIT:  # event from '[X]'
        #         running = False
        if ret:
            cv2.imshow(lbl, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
        # _, frame = camera.read()
    pygame.quit()
    camera.release()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    run_camera()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import cv2
import numpy as np
import math
from math import cos, sin, sqrt
from random import random, randint
import pygame

# set up screen and center
SCREEN_SIZE = WIDTH, HEIGHT = 1080, 720
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

BASE_COLOR = pygame.color.Color(68, 10, 103)
BASE_COLOR_H, BASE_COLOR_S, BASE_COLOR_V, BASE_COLOR_A = BASE_COLOR.hsva

WORLD_SIZE = 1000
DISTANCE_TO_VIEWING_PLANE = 200

FONT = None

mouse_x, mouse_y = (0, 0)
is_space_pressed = False

speed = 0
spaceship_angle = 0

screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()
FPS = 60

stars_positions = []
old_stars_positions = []

draw_clock = pygame.time.Clock()

FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
RECOGNITION_FRAME_WIDTH = 300
RECOGNITION_FRAME_HEIGHT = FRAME_HEIGHT / 2


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)




def perspective_transform(x, y, z):
    x_plane = 0 if z * x == 0 else DISTANCE_TO_VIEWING_PLANE / z * x
    y_plane = 0 if z * x == 0 else DISTANCE_TO_VIEWING_PLANE / z * y
    return x_plane, y_plane


def to_center(x, y):
    return x + WIDTH // 2, y + HEIGHT // 2


def rotation(x, y, angle):
    x1 = x * cos(angle) - y * sin(angle)
    y1 = x * sin(angle) + y * cos(angle)
    return x1, y1


def draw(screen):
    global delta_time

    screen.fill((0, 0, 0))

    for i in range(len(stars_positions)):
        star_xyz = stars_positions[i]

        color = pygame.color.Color(0, 0, 0)
        v = 100 * (1 - (star_xyz[2] / WORLD_SIZE))
        v = 100 if v > 100 else v
        v = 0 if v < 0 else v
        color.hsva = BASE_COLOR_H, BASE_COLOR_S, v, BASE_COLOR_A

        star_screen_x, star_screen_y = to_center(*perspective_transform(*star_xyz))
        pygame.draw.circle(screen, color, (int(star_screen_x), int(star_screen_y)), 2)
        old_star_screen_x, old_star_screen_y = to_center(*perspective_transform(*old_stars_positions[i]))

        if abs(old_stars_positions[i][2] - stars_positions[i][2]) < 30:
            pygame.draw.line(screen, color, (old_star_screen_x, old_star_screen_y), (star_screen_x, star_screen_y), 2)

    # Crosshair
    pygame.draw.circle(screen, (255, 255, 255), to_center(0, 0), 20, 2)
    pygame.draw.line(screen, (255, 255, 255), to_center(0, -30), to_center(0, +30), 2)
    pygame.draw.line(screen, (255, 255, 255), to_center(-30, 0), to_center(+30, 0), 2)

    debug_text = FONT.render(f'speed: {speed}, angle: {spaceship_angle}', 1, (255, 255, 255))
    screen.blit(debug_text, (5, 5))


def update():
    global speed, spaceship_angle

    if is_space_pressed:
        speed = speed * 2

    # spaceship_angle = ((WIDTH // 2) - mouse_x) / 10000

    if is_space_pressed:
        spaceship_angle = spaceship_angle * 2

    for i in range(len(stars_positions)):
        old_stars_positions[i] = stars_positions[i]
        x, y, z = stars_positions[i]

        x, y = rotation(x, y, spaceship_angle)
        z -= speed

        if z < 1:
            z = WORLD_SIZE

        if z > WORLD_SIZE:
            z = 1

        stars_positions[i] = x, y, z
def init():
    global FONT

    pygame.init()
    FONT = pygame.font.Font(None, 22)

    # generating 200 stars
    for i in range(2000):
        x, y, z = randint(-WORLD_SIZE, WORLD_SIZE), \
                  randint(-WORLD_SIZE, WORLD_SIZE), \
                  randint(1, WORLD_SIZE)
        stars_positions.append((x, y, z))
        old_stars_positions.append((x, y, z))

    # pygame.mixer.music.load('eduard-artemev-polet.mp3')
    # pygame.mixer.music.play()
init()


while (1):
    events = pygame.event.get()
    for event in events:
        # при закрытии окна
        # when the window is closed
        if event.type == pygame.QUIT:
            running = False
        # elif event.type == pygame.MOUSEMOTION:
        #     mouse_x, mouse_y = event.pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_space_pressed = True
            if event.key == pygame.K_UP:
                # Change color
                BASE_COLOR = pygame.color.Color(randint(0, 255), randint(0, 255), randint(0, 255))
                BASE_COLOR_H, BASE_COLOR_S, BASE_COLOR_V, BASE_COLOR_A = BASE_COLOR.hsva
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                is_space_pressed = False

    # отрисовка объектов
    # drawing objects
    draw(screen)

    # изменение свойств объектов
    # changing properties of objects
    update()

    # пауза на 1 / FPS cek
    # pause for 1 / FPS cek
    clock.tick(FPS)

    # обновление экрана
    # screen refresh
    # update entire screen
    pygame.display.flip()

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        # 2 Rectangles for left hand and right hand
        cv2.rectangle(frame, (0, 200), (RECOGNITION_FRAME_WIDTH, 500), (0, 255, 0), 0)
        cv2.rectangle(frame, (FRAME_WIDTH - RECOGNITION_FRAME_WIDTH, 200), (FRAME_WIDTH, 500), (0, 255, 0), 0)

        # Area to for recognition
        left_recog_area = frame[200:500, 0:300]
        right_recog_area = frame[200:500, 980:1280]

        hsv = cv2.cvtColor(left_recog_area, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(right_recog_area, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)


        # extract skin colur imagw
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask2 = cv2.inRange(hsv2, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask2 = cv2.dilate(mask2, kernel, iterations=4)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        mask2 = cv2.GaussianBlur(mask2, (5, 5), 100)

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # find contour of max area(hand)
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        contour2 = max(contours2, key=lambda x: cv2.contourArea(x))

        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        epsilon2 = 0.0005 * cv2.arcLength(contour2, True)
        approx2 = cv2.approxPolyDP(contour2, epsilon2, True)

        # make convex hull around hand
        hull = cv2.convexHull(contour)
        hull2 = cv2.convexHull(contour2)

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacontour = cv2.contourArea(contour)
        areahull2 = cv2.contourArea(hull2)
        areacontour2 = cv2.contourArea(contour2)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacontour) / areacontour) * 100
        arearatio2 = ((areahull2 - areacontour2) / areacontour2) * 100

        # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        hull2 = cv2.convexHull(approx2, returnPoints=False)
        defects2 = cv2.convexityDefects(approx2, hull2)

        # l = no. of defects
        l = 0

        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(left_recog_area, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(left_recog_area, start, end, [0, 255, 0], 2)

        l += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacontour < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'Best of luck', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 3:

            if arearatio < 27:
                cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)



        '''Right Hand'''
        # l = no. of defects
        l2 = 0

        # code for finding no. of defects due to fingers
        for i2 in range(defects2.shape[0]):
            s2, e2, f2, d2 = defects2[i2, 0]
            start2 = tuple(approx2[s2][0])
            end2 = tuple(approx2[e2][0])
            far2 = tuple(approx2[f2][0])
            pt2 = (100, 180)

            # find length of all sides of triangle
            a2 = math.sqrt((end2[0] - start2[0]) ** 2 + (end2[1] - start2[1]) ** 2)
            b2 = math.sqrt((far2[0] - start2[0]) ** 2 + (far2[1] - start2[1]) ** 2)
            c2 = math.sqrt((end2[0] - far2[0]) ** 2 + (end2[1] - far2[1]) ** 2)
            s2 = (a2 + b2 + c2) / 2
            ar2 = math.sqrt(s2 * (s2 - a2) * (s2 - b2) * (s2 - c2))

            # distance between point and convex hull
            d2 = (22 * ar2) / a2

            # apply cosine rule here
            angle2 = math.acos((b2 ** 2 + c2 ** 2 - a2 ** 2) / (2 * b2 * c2)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle2 <= 90 and d2 > 30:
                l2 += 1
                cv2.circle(right_recog_area, far2, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(right_recog_area, start2, end2, [0, 255, 0], 2)

        l2 += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l2 == 1:
            if areacontour2 < 2000:
                cv2.putText(frame, 'Put hand in the box', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio2 < 12:
                    cv2.putText(frame, '0', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio2 < 17.5:
                    cv2.putText(frame, 'Best of luck', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                else:
                    cv2.putText(frame, '1', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l2 == 2:
            cv2.putText(frame, '2', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l2 == 3:

            if arearatio2 < 27:
                cv2.putText(frame, '3', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'ok', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l2 == 4:
            cv2.putText(frame, '4', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l2 == 5:
            cv2.putText(frame, '5', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l2 == 6:
            cv2.putText(frame, 'reposition', (600, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        if (l == 1 or l2 == 1) and (areacontour > 2000 and areacontour2 > 2000):
            if l == 1 and l2 == 1:
                speed = 20
            elif l == 1:
                spaceship_angle = -0.05
            elif l2 == 1:
                spaceship_angle = 0.05
        else:
            speed = 0


        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('mask2', mask2)
        cv2.imshow('frame', frame)


    except Exception as e:
        print(e)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()



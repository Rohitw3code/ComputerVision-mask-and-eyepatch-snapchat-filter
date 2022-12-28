import cv2
import mediapipe as mp
import math
import random as rd
import numpy as np
from cvzone.HandTrackingModule import HandDetector

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class Tool():
    def __init__(self, toolImage, x, y, name="unknown", border=True, replaceImage=None):
        self.name = name
        self.toolImage = toolImage
        self.originalImage = toolImage
        self.x = x
        self.y = y
        self.border = border
        self.replaceImage = replaceImage
        self.selected = False
        self.desired_size_x = 40
        self.desired_size_y = 40
        self.ccode = [(249, 207, 136),(0,0,0), (249, 248, 212), (100, 100, 100), (166, 142, 232), (146, 175, 224)]
        # self.desired_size_x, self.desired_size_y = self.toolImage.shape[:2]

    def setImage(self, img, border=True):
        self.toolImage = cv2.resize(self.toolImage, (self.desired_size_x, self.desired_size_y))
        img[self.x:self.x + self.desired_size_x, self.y:self.y + self.desired_size_y, :] = self.toolImage[
                                                                                           :self.desired_size_x,
                                                                                           :self.desired_size_y, :]
        if border:
            self.drawBorder(img)

    def drawBorder(self, img):
        if self.selected and self.border:
            start_point = (self.y, self.x)
            end_point = (self.y + self.desired_size_y, self.x + self.desired_size_x)
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        if self.selected:
            try:
                if isinstance(self.replaceImage, np.ndarray):
                    self.toolImage = cv2.resize(self.replaceImage, (self.desired_size_x, self.desired_size_y))
            except NameError:
                pass

    def click(self, dist, cy, cx, img):
        if self.selected:
            self.drawBorder(img)
        if self.x < cx < self.x + self.desired_size_x and self.y < cy < self.y + self.desired_size_y:
            if dist < 40:
                self.selected = True
                self.drawBorder(img)
            else:
                self.toolImage = self.originalImage
                self.selected = False
        return self.selected, rd.choice(self.ccode)


mask = cv2.imread("mask.png")
patch = cv2.imread("patch.png")
filled = cv2.imread("with_outline.png")
wborder = cv2.imread("without_outline.png")
colorChanger = cv2.imread("color.png")

size = 40
margin = 3
maskTool = Tool(mask, 10, 10 + margin, name="Mask")
patchTool = Tool(patch, 10, 10 + size + margin, name="Eye Patch")
borderTool = Tool(wborder, 10, 10 + margin + size * 2, name="Heart", border=True, replaceImage=filled)
colorTool = Tool(colorChanger, 10, 10 + margin + size * 3, name="color")


class Shape():
    def __init__(self, color=(0, 0, 255)):
        self.color = color

    def createCircle(self, img, cx, cy, radius):
        cv2.circle(img, (cx, cy), radius, self.color, cv2.FILLED)


class SnapFilter():
    def __init__(self, threadColor=(255, 255, 255), outlineColor=(0, 0, 0), maskcolor=(0, 0, 0),
                 eyePatchColor=(0, 0, 0)):
        self.threadColor = threadColor
        self.outlineColor = outlineColor
        self.maskColor = maskcolor
        self.eyePatchColor = eyePatchColor

    def drawLips(self, img, coodinates, thickness=1, color=(0, 0, 255), innerColor=(0, 0, 0)):
        outerLips = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
        innerLips = [62, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
                     62]
        coodinates = np.array(coodinates).reshape((-1, 1, 2))
        try:
            lipsCoodinates = np.array([coodinates[l] for l in outerLips + innerLips]).reshape(-1, 1, 2)
            outerLipsCoodinates = np.array([coodinates[l] for l in outerLips]).reshape(-1, 1, 2)
            innerLipsCoodinates = np.array([coodinates[l] for l in innerLips]).reshape(-1, 1, 2)
            cv2.fillPoly(img, [lipsCoodinates], color)

            cv2.polylines(img, [outerLipsCoodinates], True, self.outlineColor, thickness)
            cv2.polylines(img, [innerLipsCoodinates], True, innerColor, thickness)
        except:
            print("Lips Not found Error")

    def drawTika(self, img, coodinates, color=(0, 0, 255)):
        path = [10, 108, 107, 9, 336, 337, 10]
        startPath = 10
        endPath = 151
        try:
            cv2.line(img, (coodinates[startPath][0], coodinates[startPath][1]),
                     (coodinates[endPath][0], coodinates[endPath][1]),
                     color, 5)
        except:
            pass

    def drawGlasses(self, img, coodinates, color=(0, 0, 0), outlineColor=(0, 0, 0), thickness=1):
        glassRight = [70, 31, 188, 55, 70]
        coodinates = np.array(coodinates).reshape((-1, 1, 2))
        try:
            glassRightCoodinates = np.array([coodinates[l] for l in glassRight]).reshape(-1, 1, 2)
            cv2.fillPoly(img, [glassRightCoodinates], color)
            cv2.polylines(img, [glassRightCoodinates], True, outlineColor, thickness)
        except:
            pass

    def drawMask(self, img, coodinates, thickness=1):
        path = [93, 137, 123, 50, 36, 49, 220, 45, 4, 275, 440, 279, 266, 280, 352, 323, 361, 288, 397, 365, 379, 378,
                400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93]
        coodinates = np.array(coodinates).reshape((-1, 1, 2))
        try:
            maskCoodinates = np.array([coodinates[l] for l in path]).reshape(-1, 1, 2)
            cv2.fillPoly(img, [maskCoodinates], self.maskColor)
            cv2.polylines(img, [maskCoodinates], True, self.outlineColor, thickness)
        except:
            pass

    def drawEyePatch(self, img, coodinates, thickness=1):
        rightPatch = [70, 53, 53, 65, 55, 193, 122, 188, 114, 120, 119, 118, 117, 111, 35, 156]
        thread = [193, 285, 295, 293, 251, 251, 293, 295, 285, 193]
        coodinates = np.array(coodinates).reshape((-1, 1, 2))
        try:
            rightPatchCoodinates = np.array([coodinates[l] for l in rightPatch]).reshape(-1, 1, 2)
            threadCoodinates = np.array([coodinates[l] for l in thread]).reshape(-1, 1, 2)
            cv2.fillPoly(img, [rightPatchCoodinates], self.eyePatchColor)
            cv2.polylines(img, [rightPatchCoodinates], True, self.outlineColor, thickness)
            cv2.polylines(img, [threadCoodinates], True, self.threadColor, thickness)
        except:
            pass

    def updateMaskColor(self, color):
        self.maskColor = color

    def updateEyePatchColor(self, color):
        self.eyePatchColor = color

    def updateOutLineColor(self, color):
        self.outlineColor = color

    def updateThreadColor(self, color):
        self.threadColor = color


def findDistance(p1, p2, img=None):
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)
    if img is not None:
        # cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return length, info, img
    else:
        return length, info


detector = HandDetector(maxHands=1, detectionCon=0.8)
handPointer = Shape()

sf = SnapFilter()
DRAW_EYE_PATCH = False
DRAW_MASK = False
OUTLINE = False

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        hand = detector.findHands(image, draw=False)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = []
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                for landmark in face.landmark:
                    x = landmark.x
                    y = landmark.y

                    shape = image.shape
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    landmarks.append([relative_x, relative_y])
        if hand:
            lmList = hand[0]['lmList']

            p1 = lmList[8][0], lmList[8][1]
            p2 = lmList[12][0], lmList[12][1]

            # handPointer.createCircle(image,cx,cy,10)
            dist, info = findDistance(p1, p2, image)[:2]
            cx, cy = info[4:]
            DRAW_MASK = maskTool.click(dist, cx, cy, image)[0]
            DRAW_EYE_PATCH = patchTool.click(dist, cx, cy, image)[0]
            OUTLINE = borderTool.click(dist, cx, cy, image)[0]
            colorInfo = colorTool.click(dist, cx, cy, image)
            if colorInfo[0]:
                colorTool.selected = False
                sf.updateMaskColor(colorInfo[1])
                sf.updateEyePatchColor(colorInfo[1])

        if DRAW_MASK:
            sf.drawMask(img=image, coodinates=landmarks)
        if DRAW_EYE_PATCH:
            sf.drawEyePatch(img=image, coodinates=landmarks)
        if OUTLINE:
            sf.outlineColor = (255, 255, 255)
        else:
            sf.outlineColor = (0, 0, 0)

        borderTool.setImage(image)
        maskTool.setImage(image)
        patchTool.setImage(image)
        colorTool.setImage(image)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()

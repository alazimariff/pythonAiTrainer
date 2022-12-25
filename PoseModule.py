import mediapipe as mp
import math
import cv2

class poseDetector():
    def __init__(self,
                 staticMode=False,
                 model_complexity=False,
                 smooth=True,
                 minDetectionCon=0.5,
                 minTrackCon=0.5):

        self.staticMode = staticMode
        self.modelComplexity = model_complexity
        self.smooth = smooth
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.staticMode, model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.minDetectionCon,
                                     min_tracking_confidence=self.minTrackCon)
        self.drawLandmarkSpec = self.mpDraw.DrawingSpec(
            thickness=2, circle_radius=2, color=(255,0,0))
        self.drawConnectionSpec = self.mpDraw.DrawingSpec(
            thickness=2, color=(34,247,10))


    def find_Person(self, frame, draw=True):
        self.imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS, self.drawLandmarkSpec, self.drawConnectionSpec)
        return frame

    def find_landmarks(self, frame, draw=True):
        self.landmark_list=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.landmark_list

    def find_angle(self, frame, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        print("ANGLE")
        print(angle)

        # Draw
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
            cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 5)
            cv2.circle(frame, (x1, y1), 11, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 16, (255, 60, 0), 2)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 16, (255, 60, 0), 2)
            cv2.circle(frame, (x3, y3), 11, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 16, (255, 60, 0), 2)

            cv2.putText(frame, str(int(angle)), (x3 - 50, y3 + 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        return angle

def main():
    cap = cv2.VideoCapture('TrainerData/curls.mp4')
    detector = poseDetector()
    while True:
        success, frame = cap.read()
        #frame = cv2.imread("TrainerData/bicep_curls.jpeg")
        frame = detector.find_Person(frame)
        landmark_list = detector.find_landmarks(frame, draw=True)
        print(landmark_list)
        if len(landmark_list) != 0:
            print(landmark_list[16])
            cv2.circle(
                frame, (landmark_list[16][1], landmark_list[16][2]), 15, (0, 0, 255), cv2.FILLED)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()
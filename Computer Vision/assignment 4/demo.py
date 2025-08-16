import cv2
from tracking import MotionTracker
from tracking import KalmanFilter
import skvideo.io

if __name__ == "__main__":

    filepath = input("Enter file path of video: ")

    frames = skvideo.io.vread(filepath)

    # KalmanFilter(dt)
    KF = KalmanFilter(0.1)
    #  MotionDetector(a, motion_thresh, dist_thresh, s, N, KalmanFilter)
    motionDetector = MotionTracker(1,1,100,1,25,KF)
    for i in range(0,len(frames)-2):
        motionDetector.update(frames,i)
        frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        cv2.imshow('image', frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.waitKey(1)
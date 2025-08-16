# Long Nguyen
# 1001705873

import sys
import numpy as np
import cv2
import argparse
from PySide6 import QtCore, QtWidgets, QtGui
from skvideo.io import vread

from tracking import KalmanFilter, MotionTracker

class VideoTracker(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames
        self.total_frames = len(frames)
        self.current_frame = 0

        self.KF = KalmanFilter(0.1)
        self.motionDetector = MotionTracker(1, 1, 100, 1, 25, self.KF)
        self.trail_history = {}

        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.total_frames - 1)

        self.next_btn = QtWidgets.QPushButton("Next Frame")
        self.prev_btn = QtWidgets.QPushButton("Previous Frame")
        self.skip_fwd_btn = QtWidgets.QPushButton("+60 Frames")
        self.skip_back_btn = QtWidgets.QPushButton("-60 Frames")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.img_label)
        layout.addWidget(self.frame_slider)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        btn_layout.addWidget(self.skip_back_btn)
        btn_layout.addWidget(self.skip_fwd_btn)
        layout.addLayout(btn_layout)

        self.frame_slider.sliderMoved.connect(self.on_slider_move)
        self.next_btn.clicked.connect(self.next_frame)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.skip_fwd_btn.clicked.connect(self.skip_forward)
        self.skip_back_btn.clicked.connect(self.skip_backward)

        self.display_frame()

    def process_frame(self, frame_index):
        self.KF = KalmanFilter(0.1)
        self.motionDetector = MotionTracker(1, 1, 100, 1, 25, self.KF)
        self.trail_history.clear()

        for i in range(frame_index + 1):
            bboxes = self.motionDetector.update(self.frames, i)
            if bboxes:
                for idx, bbox in enumerate(bboxes):
                    center = ((bbox[1] + bbox[3]) // 2, (bbox[0] + bbox[2]) // 2)
                    if idx not in self.trail_history:
                        self.trail_history[idx] = []
                    self.trail_history[idx].append(center)

        frame = self.frames[frame_index].copy()
        for trail in self.trail_history.values():
            for point in trail:
                cv2.circle(frame, (point[0], point[1]), 3, (0, 255, 0), -1)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def display_frame(self):
        rgb_frame = self.process_frame(self.current_frame)
        h, w, c = rgb_frame.shape
        img = QtGui.QImage(rgb_frame.data, w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        self.frame_slider.setValue(self.current_frame)

    def on_slider_move(self, pos):
        self.current_frame = pos
        self.display_frame()

    def next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.display_frame()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.display_frame()

    def skip_forward(self):
        self.current_frame = min(self.current_frame + 60, self.total_frames - 1)
        self.display_frame()

    def skip_backward(self):
        self.current_frame = max(self.current_frame - 60, 0)
        self.display_frame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Tracker GUI")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--grey", type=bool, default=False)
    args = parser.parse_args()

    if args.num_frames > 0:
        frames = vread(args.video_path, num_frames=args.num_frames, as_grey=args.grey)
    else:
        frames = vread(args.video_path, as_grey=args.grey)

    print("[INFO] Loaded video with shape:", frames.shape)
    if args.grey:
        frames = np.stack((frames,) * 3, axis=-1)  # Convert grayscale to RGB

    app = QtWidgets.QApplication([])
    window = VideoTracker(frames)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())
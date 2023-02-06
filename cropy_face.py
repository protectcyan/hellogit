import os
import pickle
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import imageio
import time
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from numpy.linalg import norm

FRATE_RATE = 25
w_thresh = 140
h_thresh = 140
crop_size = 512
half_crop_size = crop_size // 2
videofile = Path(r"")
save_path = Path('./face/video_crop') / videofile.name[:-4]
os.makedirs(str(save_path), exist_ok=True)
def extend_rect(face_bbox):
    x1, y1, x2, y2 = face_bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_h = max((y2 - y1)*3//4, (x2 - x1)*3//4)
    half_w = half_h
    new_x1 = cx - half_w
    new_y1 = cy - half_h
    new_x2 = cx + half_w
    new_y2 = cy + half_h
    return (new_x1, new_y1, new_x2, new_y2)
def crop_face(img, face_bbox):
    x1, y1, x2, y2 = map(int, extend_rect(face_bbox))
    img_pad = np.pad(img, ((int(max(0, -y1)), int(max(0, y2 - frame_h))), (int(max(0, -x1)), int(max(0, x2 - frame_w))), (int(0), int(0))),mode= 'constant')
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame_w)
    y2 = min(y2, frame_h)
    h, w = x2 - x1, y2 - y1
    if h < h_thresh or w < w_thresh:
        return None
    if (img_pad.shape[2] != 3):
        print(x1, y1, x2, y2)
        print(int(max(0, -y1)), int(max(0, y2 - frame_h)))
        breakpoint()
    out = img_pad[y1:y2, x1:x2]
    if out.shape[0] < 10 or out.shape[1] < 10:
        print(x1, y1, x2, y2)
        breakpoint
    return out

app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(354, 354))

cap = cv2.VideoCapture(str(videofile))
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
cnt = 0
skip_frames = FRATE_RATE * 90
# cv2.namedWindow("frame")
from time import time
print('跳过片头')
ret, img = cap.read()
frame_h, frame_w, _ = img.shape
for i in range(skip_frames):
    if cap.isOpened():
        cap.grab()
    else:
        break
skip_cnt = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    if skip_cnt < FRATE_RATE:
        skip_cnt += 1
        cap.grab()
        continue
    skip_cnt = 0
    ret, frame = cap.read()
    if ret == True:
        cnt += 1   
        if cnt < 41:
            continue
        t1 = time()
        frame_to_show = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', frame_to_show)
        cv2.setWindowTitle('frame', f'{cnt}')
        key = cv2.waitKey(4)
        if key == ord('q'):
            break
        frame = frame[:int(frame_h*0.9)]
        faces = app.get(frame)
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            face_img = crop_face(frame, (x1, y1, x2, y2))
            if face_img is not None:
                save_file = str(save_path / f'{cnt}_{i}.jpg')
                print(face.keys())
                print(f'{cnt}_{i} detection_score:{face.get("det_score")}')
                imageio.imsave(save_file, face_img[:,:,::-1])
                with open(save_file[:-4]+'.pkl', 'w') as fout:
                    pickle.dump(face, fout)
        # ret = cv2.imwrite(save_file, frame)
        # print(ret)
    else: 
        break
# img = cv2.imread('face/images/w.jpg')
# faces = app.get(img)
# face1 = faces[0]
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

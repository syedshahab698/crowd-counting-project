import os
import os.path as osp
import cv2
from PIL import Image
import torch

def results_parser(results):
  s = ""
  if results.pred[0].shape[0]:
    for c in results.pred[0][:, -1].unique():
      n = (results.pred[0][:, -1] == c).sum()  # detections per class
      s += f"{n} {results.names[int(c)]}{'s' * (n > 1)}, "  # add to string
  return s

def runVideo(model, video):
    video_name = osp.basename(video)
    outputpath = osp.join('data/video_output', video_name)

    # Create A Dir to save Video Frames
    os.makedirs('data/video_frames', exist_ok=True)
    frames_dir = osp.join('data/video_frames', video_name)
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    frame_count = 0
    
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(frame)
        print(results_parser(result))
        result.render()
        image = Image.fromarray(result.ims[0])
        
        image.save(osp.join(frames_dir, f'{frame_count}.jpg'))
    cap.release()
    # convert frames in dir to a single video file without using ffmeg
    image_folder = frames_dir
    video_name = outputpath

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

CFG_MODEL_PATH = "models/yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=CFG_MODEL_PATH, force_reload=True, device=0)
runVideo(model, "persons_walking_video.mp4")

# Libraries and Modules
from mylib.centroidtraker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread

# Sending info
def send_mail():
    Mailer.send(config.MAIL)

# Counter
def people_counter():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required = False,
                    help = "path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required = True,
                    help = "path to pre-trained model")
    ap.add_argumnet("-i", "--input", type = str,
                    help = "path to optional input video file")
    ap.add_argument("-o", "--output", typr = str,
                    help = "path to optional output video file")
    
    if not args.get("input",False):
        print("[INFO] Starting the live stream..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)

    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])
import json
import os
import shutil
import cv2
from flask import Flask, request, Response, jsonify
from werkzeug.utils import secure_filename

from face_detect.face_detector import FaceDetector
from video_process import VideoCamera

app = Flask(__name__)

if not os.path.isdir('./videos'):
    os.mkdir('./videos')


# Video upload API
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        global f, url
        url = None
        f = request.files['video']
        f.save('./videos/' + f.filename)
        return json.dumps(True)


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            return "video not avilable"
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# video stream API
@app.route('/video_feed')
def video_feed():
    name = request.args.get('name')
    video_process = VideoCamera(video='./videos/' + str(name))
    return Response(gen(video_process),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Black list people data push api
@app.route('/api/uploadImage', methods=['GET', 'POST'])
def upload_images():
    fd = FaceDetector('face_detect/weight/model.pb')
    name = request.args.get('name')

    path = './BlackListed_people'
    temp_path = os.path.join(path, str(name))

    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)
    else:
        shutil.rmtree(temp_path)
        os.mkdir(temp_path)

    if request.method == 'POST':
        files = request.files.getlist("images")

        try:
            for f in files:
                full_path = os.path.join(temp_path, str(secure_filename(f.filename)))
                f.save(full_path)

        except FileExistsError as fee:
            return jsonify({'message': 'image file not exist'})

    for f in os.listdir(temp_path):
        im_path = os.path.join(temp_path, f)
        img = cv2.imread(im_path)
        boxes, scores = fd(img)
        if len(boxes) == 1:
            box = list(boxes)[0]
            ymin, xmin, ymax, xmax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped_img = img[ymin:ymax, xmin:xmax]
            cv2.imwrite(im_path, cropped_img)
        else:
            return jsonify({'message': 'No face detect or many face detect'})

    return jsonify({'message': 'images uploaded!'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

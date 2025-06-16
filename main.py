from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        results = camera.get_frame()
        frame = results[0]
        if not frame:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting posture analysis web app with MediaPipe")
    print("Access the app at http://localhost:5000")
    app.run(host='0.0.0.0', debug=True)

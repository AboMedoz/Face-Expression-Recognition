import os

from flask import Flask, render_template, Response

from main import predict_expr

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
TEMPLATES = os.path.join(ROOT, 'templates')

app = Flask(__name__, template_folder=TEMPLATES)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(predict_expr(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
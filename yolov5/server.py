import time

from flask import request, Flask
import base64
import cv2
import numpy as np
# from mask import Infer
from Detectbase.PersonInfer import PersonInfer
from Detectbase.MaskInfer import MaskInfer
from Detectbase.PoseInfer import PoseInfer
import json

# import

app = Flask(__name__)

with open("config.json", 'r') as c:
    config = json.load(c)

infer = {}
for id, task in enumerate(config['task_list']):
    if task == "person":
        infer[task] = PersonInfer(config["model"][task])
    elif task == "pose":
        infer[task] = PoseInfer(config["model"][task])
    elif task == "mask":
        infer[task] = MaskInfer(config["model"][task])
    # elif task == "smoke":
    #     infer[task] = Detectbase.SmokeInfer(config["model"][task])
    # elif task == "alarm":
    #     infer[task] = Detectbase.AlarmInfer(config["model"][task])
    else:
        print("%s infer is not included in plan!" %(task))
@app.route('/')
@app.route("/detect", methods=['POST', 'GET'])
def detect():
    # 解析图片数据
    img = base64.b64decode(str(request.form['image']))
    task_list = request.form['task'].split(',')
    image_data = np.fromstring(img, np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    res = {}
    print(task_list)
    for task in task_list:
        print("Current infer module:%s" %(task))
        start = time.time()
        infer[task].run(image_data,res)
        print(time.time() - start)
        res[task] = infer[task].res
    return (str(res))

    # checkmask.run(image_data)
    # return (str(res))


# def predict():
#     # 解析图片数据
#     start = time.time()
#     # print(type(request.form['image']))
#     # img = base64.b64decode(request.form['image'])
#     print(type(request.form['image']))
#     # print(request.form['image'])
#     image_data = np.fromstring(request.form['image'], dtype=int)
#     print(len(request.form['image']))
#     print(len(image_data))
#     image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
#     # print(type(request.form['image']))
#     image_data.imwrite('../dataset/test/a.jpg',image_data)
#     print(time.time() - start)
#
#     start = time.time()
#     checkmask.run(image_data)
#     print(time.time() - start)
#     return (str(checkmask.res))


if __name__ == "__main__":
    app.run("127.0.0.1", port=5005)

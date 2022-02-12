import requests
import base64
import time
#将图片数据转成base64格式
img_path = "/data/wangyj/02_Study/Pytorch/dataset/test/Personcheck.png"
with open(img_path, 'rb') as f:
    img = base64.b64encode(f.read()).decode()
image = []
image.append(img)
res = {"image":image,"task":"person,pose,mask"}
#访问服务
start = time.time()
response = requests.post("http://127.0.0.1:5005/detect",data=res)
print(time.time()-start)
print(response.text)

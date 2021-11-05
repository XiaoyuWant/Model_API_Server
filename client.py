# -*- coding: utf-8 -*-
import requests
url="http://127.0.0.1:8080/predict"

files = {'file':('image.jpg',open("images/7.jpg",'rb'),' image/jpeg')}
r = requests.post(url,files=files)

print(r.text)

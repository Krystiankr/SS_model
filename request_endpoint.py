import requests
import sys

url = "http://192.168.0.11:8000/predict/"

payload={}
image_path = sys.argv[1]
files=[
  ('file',('resized_image_2.bmp',open(image_path ,'rb'),'image/bmp'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)

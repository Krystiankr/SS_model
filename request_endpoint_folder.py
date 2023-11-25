import requests
import sys
import os

url = "http://192.168.1.59:8000/predict/"

root_dir = sys.argv[1]
sub_folders = os.listdir(root_dir)

for _sub_folder in sub_folders:
  sub_dir = os.listdir(f"{root_dir}/{_sub_folder}")
  print(f"Checking subfolder: {_sub_folder.upper()}...")
  for _file in sub_dir:
    payload={}
    image_path = f"{root_dir}/{_sub_folder}/{_file}"
    files=[
      ('file',('resized_image_2.bmp',open(image_path ,'rb'),'image/bmp'))
    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(f"File name: {_file}")
    print(response.text)
  print('-' * 10)

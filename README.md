# Yolov3_flask
Object detection on the Web side using Yolov3(PyTorch) and Flask

# Introduction
This project is a little demo for web object detection, which include yolov3 object detection, flask and html

The main idea include YOLOV3 and Flask
Yolov3 comes form ultralytics, you can use their project to train one model which satisfy your purpose

# Modification
1.Modify the original detect.py to detect_for_flask.py, provide one interface for Flask

2.All uploaded file will be renamed by time and saved to the "upload_files" folder

3.Detected image will be saved to the "output" folder


# Qucik start

1. Follow the ult-yolov3 requirements to config the environments
2. Download or train one model, put ".weights/.pt" file to the weights folder, and config the right cfg,other config can be set on opt 
3. Start serve.py, then input "http://127.0.0.1:2222/upload" on the website, upload the picture, then you will get the result and detection informations.


# Citation
https://github.com/ultralytics/yolov3 
https://blog.csdn.net/rain2211/article/details/105965313/





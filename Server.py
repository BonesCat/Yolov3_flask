import time
import os
# 导入flask库中的Flask类与request对象
from flask import Flask, request, flash, redirect, render_template, jsonify
from datetime import timedelta

# 导入模型相关函数
from detect_for_flask import *


app = Flask(__name__)

# 设置上传文件的保存位置
UPLOAD_FOLDER = 'upload_files'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# 配置路径到app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=5) # timedalte 是datetime中的一个对象，该对象表示两个时间的差值

print("SEND_FILE_MAX_AGE_DEFAULT:", app.config['SEND_FILE_MAX_AGE_DEFAULT'])

# 预先初始化模型
model_inited, opt = init_model()

# 处理文件名的有效性
def allow_filename(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST']) # 添加路由

def upload():
    if request.method == 'POST':
        # 如果上传的file不是在files
        if 'file' not in request.files:
            # Flask 消息闪现
            flash('not file part!')
            # 重新显示当前url页面
            return  redirect(request.url)

        '''
        Flask 框架中的 request 对象保存了一次HTTP请求的一切信息。
        files 记录了请求上传的文件
        '''
        f = request.files['file']

        # 处理空文件
        if f.filename == '':
            flash("Nothing file upload")
            return redirect(request.url)

        # 文件非空，且格式满足
        if f and allow_filename(f.filename):
            # 保存上传文件至本地
            # 按照格式获取当前时间,从命名文件
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            file_extension = f.filename.split('.')[-1]
            new_filename = now + '.' + file_extension
            file_path = './' + app.config['UPLOAD_FOLDER'] + '/' + new_filename
            f.save(file_path)

            # 进行预测，并显示图片
            img, obj_infos = detect(model_inited, opt, file_path)
            return render_template('upload_ok.html', det_result = obj_infos)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2222)











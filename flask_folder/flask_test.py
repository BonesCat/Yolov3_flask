from flask import request, Flask

# 创建引用实例
app = Flask(__name__)

# 视图函数
@app.route('/')
def index():
    return '<h1> Hello Flask!<h1>'

# 启动实施
if __name__ == '__main__':
    app.run()


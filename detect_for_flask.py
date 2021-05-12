import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

'''
根据原始YoloV3中的detect.py，重写了检测函数，来适配flask
'''


def init_model():
    '''
    模型参数初始化
    ：无输入参数
    :return: 完成初始的模型 和 opt设置
    '''
    # paraments config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='weights path')
    parser.add_argument('--output', type=str, default='output', help='output folder')  # detect result will be saved here
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    # init paraments
    out, weights, save_txt = opt.output, opt.weights, opt.save_txt

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, opt.img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    return model, opt

def detect(model, opt, image_path):
    '''
    :param model: 完成初始化的模型
    :param opt: opt参数
    :param image_path:传入的图片地址 
    :param save_img: 是否保存图片
    :return: 完成定位后的结果
    '''
    # Eval mode
    model.to(opt.device).eval()
    # Save img?
    save_img = True

    # Process the upload image

    # read img
    img0 = cv2.imread(image_path)  # BGR
    assert img0 is not None, 'Image Not Found ' + image_path

    # Padded resize
    img = letterbox(img0, new_shape=opt.img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(opt.device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()
        # print("pred:", pred)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # 这是检测出来的所有object的，检测结果是一个二维list
            # 每一行存放的是一个obj的左上，右下四个坐标，置信度，类别
            # print("det", det)

            p, s = image_path, ''

            save_path = str(Path(opt.output) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            # 若检测出了对象，则list不为空
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # 设置字典，写入每个目标数据
                obj_info_list = []
                # 遍历二维det中的每行，从而对每一个obj进行处理
                # Write results
                for *xyxy, conf, cls in det:
                    if opt.save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=colors[int(cls)]) # 参数xyxy中包含着bbox的坐标
                    # 记录单个目标的坐标，类别，置信度
                    sig_obj_info =('%s %g %g %g %g %g' ) % (names[int(cls)], *xyxy, conf)
                    print("sig_obj_info:", sig_obj_info)
                    obj_info_list.append(sig_obj_info)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))


            # Save results (image with detections)
            if save_img:
                # 两次保存
                # 1.永久保存检测结果，存入output文件夹
                cv2.imwrite(save_path, img0)
                # 2.暂存文件，用于显示
                cv2.imwrite('./static/temp.jpg', img0)

    print('Done. (%.3fs)' % (time.time() - t0))
    return img0, obj_info_list


if __name__ == '__main__':
    img_path = './data/samples/timg1.jpg'
    model_inited, opt = init_model()
    result,obj_infos = detect(model = model_inited, opt = opt, image_path=img_path)
    print(obj_infos)

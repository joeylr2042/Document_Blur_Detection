import paddle
import cv2
import numpy as np
from op import *
from db_postprocess import *
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='./text_blur.jpg', help='image path')
parser.add_argument('--thresh_v', type=float, default=300, help='threshold of the variance')
parser.add_argument('--thresh_d', type=float, default=0.7, help='threshold of the detection')

opt = parser.parse_args()
image_file = opt.image
threshold_v = opt.thresh_v
threshold_d = opt.thresh_d


def transform(img, ops=None):
    """ transform """
    if ops is None:
        ops = []
    data = {'image': img}
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators():
    ops = [DetResizeForTest(limit_side_len=960, limit_type='max'),
           NormalizeImage(std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406], scale=1. / 255, order='hwc'),
           ToCHWImage(),
           KeepKeys(keep_keys=['image', 'shape'])]
    return ops


def infer(img):
    img = np.expand_dims(img[0], axis=0)
    tensor_img = np.array(img, dtype=np.float32)
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: tensor_img},
                      fetch_list=fetch_targets)
    return results[0]


def get_detection(image, wide, height, write_results=False):
    image = transform(image, ops=operators)
    image = infer(image)
    boxes = db_postprocess(image, wide, height)
    if write_results:
        write_result(boxes, "text_blur.jpg")
    return boxes


def write_result(boxes, image_file):
    json_result = json.dumps(boxes.tolist())
    with open(image_file + ".res.txt", "w") as f:
        f.write(json_result)
    image = cv2.imread(image_file)
    image = draw_boxes(image, boxes)
    cv2.imwrite(image_file + ".res.png", image)


def draw_boxes(image, boxes):
    for box in boxes:
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 1)
    return image


def var_of_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_n = np.asarray(gray).astype(np.float64)
    edge_det = cv2.Laplacian(gray_n, cv2.CV_64F)
    edge_var = edge_det.var()
    return edge_var


threshold = 0.65
model_path_prefix = "ch_ppocr_mobile_v2.0_det_infer/inference"
image = cv2.imread('./text_blur.jpg')
wide, height = image.shape[1], image.shape[0]
paddle.enable_static()
exe = paddle.static.Executor(paddle.CPUPlace())
[inference_program, feed_target_names, fetch_targets] = paddle.static.load_inference_model(model_path_prefix, exe)

operators = create_operators()
db_postprocess = DBPostProcess(box_thresh=threshold_d)

edge_var = var_of_laplacian(image)
print(edge_var)
if edge_var < threshold_v:
    print("blur")
else:
    boxes = get_detection(image, wide, height, True)
    if len(boxes) <= 8:
        print("blur")
    else:
        print("clear")

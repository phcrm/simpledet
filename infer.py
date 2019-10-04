import os, argparse
import importlib
import json
import time
import cv2
import numpy as np
import mxnet as mx
from   core.detection_module import DetModule
from   utils.load_model      import load_checkpoint
from utils.patch_config import patch_config_as_nothrow


coco = (
'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

coco = (
    'tractor', 'schwader', 'anhänger', 'test'
)

class TDNDetector:
    def __init__(self, configFn, ctx, outFolder, threshold):
        os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
        config = importlib.import_module(configFn.replace('.py', '').replace('/', '.'))
        _,_,_,_,_,_, self.__pModel,_, self.__pTest, self.transform,_,_,_ = config.get_config(is_train=False)
        self.__pModel = patch_config_as_nothrow(self.__pModel)
        self.__pTest = patch_config_as_nothrow(self.__pTest)
        self.resizeParam = (800, 1200)
        if callable(self.__pTest.nms.type):
            self.__nms = self.__pTest.nms.type(self.__pTest.nms.thr)
        else:
            from operator_py.nms import py_nms_wrapper
            self.__nms = py_nms_wrapper(self.__pTest.nms.thr)
        arg_params, aux_params = load_checkpoint(self.__pTest.model.prefix, self.__pTest.model.epoch)
        sym = self.__pModel.test_symbol
        from utils.graph_optimize import merge_bn
        sym, arg_params, aux_params = merge_bn(sym, arg_params, aux_params)
        self.__mod = DetModule(sym, data_names=['data','im_info','im_id','rec_id'], context=ctx)
        self.__mod.bind(data_shapes=[('data', (1, 3, self.resizeParam[0], self.resizeParam[1])), 
                                     ('im_info', (1, 3)),
                                     ('im_id', (1,)),
                                     ('rec_id', (1,))], for_training=False)
        self.__mod.set_params(arg_params, aux_params, allow_extra=False)
        self.__saveSymbol(sym, outFolder, self.__pTest.model.prefix.split('/')[-1])
        self.__threshold = threshold

    def __call__(self, imgFilename): # detect onto image
        roi_record, scale = self.__readImg(imgFilename)
        h, w = roi_record['data'][0].shape

        im_c1 = roi_record['data'][0].reshape(1,1,h,w)
        im_c2 = roi_record['data'][1].reshape(1,1,h,w)
        im_c3 = roi_record['data'][2].reshape(1,1,h,w)
        im_data = np.concatenate((im_c1, im_c2, im_c3), axis=1)

        im_info, im_id, rec_id = [(h, w, scale)], [1], [1] 
        data = mx.io.DataBatch(data=[mx.nd.array(im_data),
                                     mx.nd.array(im_info),
                                     mx.nd.array(im_id),
                                     mx.nd.array(rec_id)])
        self.__mod.forward(data, is_train=False)
        # extract results
        outputs = self.__mod.get_outputs(merge_multi_context=False)
        rid, id, info, cls, box = [x[0].asnumpy() for x in outputs]
        rid, id, info, cls, box = rid.squeeze(), id.squeeze(), info.squeeze(), cls.squeeze(), box.squeeze()
        cls = cls[:, 1:]   # remove background
        box = box / scale
        output_record = dict(rec_id=rid, im_id=id, im_info=info, bbox_xyxy=box, cls_score=cls)
        output_record = self.__pTest.process_output([output_record], None)[0]
        final_result  = self.__do_nms(output_record)
        # obtain representable output
        detections = []
        for cid ,bbox in final_result.items():
            idx = np.where(bbox[:,-1] > self.__threshold)[0] 
            for i in idx:
                final_box = bbox[i][:4]
                score = bbox[i][-1]
                detections.append({'cls':cid, 'box':final_box, 'score':score})
        return detections,None

    def __do_nms(self, all_output):
        box   = all_output['bbox_xyxy']
        score = all_output['cls_score']
        final_dets = {}
        for cid in range(score.shape[1]):

            score_cls = score[:, cid]
            valid_inds = np.where(score_cls > self.__threshold)[0]
            box_cls = box[valid_inds]
            score_cls = score_cls[valid_inds]
            if valid_inds.shape[0]==0:
                continue
            det = np.concatenate((box_cls, score_cls.reshape(-1, 1)), axis=1).astype(np.float32)
            det = self.__nms(det)
            cls = coco[cid]
            final_dets[cls] = det
        return final_dets

    def __readImg(self, imgFilename):
        img = cv2.imread(imgFilename, cv2.IMREAD_COLOR)
        height, width, channels = img.shape
        roi_record = {'gt_bbox': np.array([[0., 0., 0., 0.]]),'gt_class': np.array([0])}
        roi_record['image_url'] = imgFilename
        roi_record['h'] = height
        roi_record['w'] = width
 
        for trans in self.transform:
            trans.apply(roi_record)
        img_shape = [roi_record['h'], roi_record['w']]
        shorts, longs = min(img_shape), max(img_shape)
        scale = min(self.resizeParam[0] / shorts, self.resizeParam[1] / longs)

        return roi_record, scale

    def __saveSymbol(self, sym, outFolder, fnPrefix):
        if not os.path.exists(outFolder): os.makedirs(outFolder)
        resFilename = os.path.join(outFolder, fnPrefix + "_symbol_test.json")
        sym.save(resFilename)

        
def draw_rectangle(img, box_cordinates, cls):
    colors = {
        'tractor': (255,0,0),
        'schwader': (0,255,0),
        'anhänger': (0,0,255),
        'test': (255,255,255)
    }
    x1, y1, x2, y2 = box_cordinates
    return cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 2)

        
import mxnet as mx
import argparse
from infer import TDNDetector
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Test Detection')
    parser.add_argument('--config', type=str, default='config/faster_r50v1_fpn_1x-mahd.py', help='config file path')
    parser.add_argument('--ctx',    type=int, default=0,     help='GPU index. Set negative value to use CPU')
    #parser.add_argument('--inputs', type=str, nargs='+', required=True, default='', help='File(-s) to test')
    parser.add_argument('--output', type=str, default='results', help='Where to store results')
    parser.add_argument('--threshold', type=float, default=0.5,  help='Detector threshold')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()   
    ctx = mx.gpu(args.ctx) if args.ctx>=0 else args.cpu()
    #imgFilenames = args.inputs
    imgFilenames = ['data/src/mahd/JPEGImages/00000007.jpg', 'data/src/mahd/JPEGImages/schwader_9.jpg']
    detector = TDNDetector(args.config, ctx, args.output, args.threshold)
    
    images_classified = []

    # predict on image
    for i, imgFilename in enumerate(imgFilenames):
        print(imgFilename)
        dets,_= detector(imgFilename)
        print(dets)

        # draw predicted boxes on image
        box_img = cv2.imread(imgFilename, cv2.IMREAD_COLOR)
        for box in dets:
            cords = box['box']
            cls = box['cls']
            box_img = draw_rectangle(box_img, cords, cls)
        images_classified.append(box_img)

    # show image
    for img in images_classified:
        cv2.imshow('box_img', img)
        time.sleep(0.025)
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
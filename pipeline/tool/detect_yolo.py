# Utilizing the following code snippet based on the algorithm proposed in the paper:
# Title: "Woodpecker: Hallucination correction for multimodal large language models"
# Author: Yin et al.
# Original code source: https://github.com/BradyFU/Woodpecker/blob/main/models/detector.py

import yaml
import torch
import os
import shortuuid
import numpy as np
from PIL import Image
from torchvision.ops import box_convert
from pipeline.tool.ocr import *
from pipeline.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from ultralytics import YOLOWorld
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class DetectModel:
    def __init__(self, config):
        self.config = config
        self.model = load_model(self.config["tool"]["detect"]["groundingdino_config"], 
                                self.config["tool"]["detect"]["model_path"], 
                                device=self.config["tool"]["detect"]["device"])
        self.yolo_model = YOLOWorld('weights/yolov8s-worldv2.pt')
        
    def IoU(self, box1, box2):
        '''
        两个框（二维）的 iou 计算
        注意：边框以左上为原点
        box:[x1,y2,x2,y2],依次为左上右下坐标
        '''
        h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
        area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
        inter = w * h
        union = area_box1 + area_box2 - inter
        iou = inter / union
        return iou
    
    def box_Union(self, bboxes1, bboxes2, cls1, cls2):
        '''
        遍历boxes1中的每个框A
        检查boxes2中的每个框B是否与A有重叠
        如果有重叠，则合并两个框，将A和B合并为一个框，并删除boxes2中的B框，添加A到results
        如果没有重叠，则将当前框类别添加到results中
        '''
        bboxes = bboxes1
        cls = cls1
        for A in bboxes1:
            for B, clsB in zip(bboxes2, cls2):
                iou = self.IoU(A, B)
                if iou > 0.9:
                    A = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]
                    bboxes2.remove(B)
                    cls2.remove(clsB)
                    break
        bboxes.extend(bboxes2)
        cls.extend(cls2)
        return bboxes, cls
    
    def draw(self, bboxes, labels, image_source, image_path,  save_path):
        save_path='/root/zrw/code/EasyDetect-main/runs/GDetect/'
        dir_name = image_path.split("/")[-1][:-4]
        cache_dir = save_path + dir_name
        print(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        image_path_list = []
        # 创建绘图对象
        fig, ax = plt.subplots(dpi=300)
        ax.imshow(image_source)
        # 绘制归一化后的边界框和标签
        for bbox, label in zip(bboxes, labels):
            x_min, y_min, x_max, y_max = bbox
            h, w, _ = image_source.shape
            x_min *= w
            y_min *= h
            x_max *= w
            y_max *= h
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, label, color='r')
        # 关闭坐标轴
        ax.axis('off')

        # 保存可视化结果
        crop_id = shortuuid.uuid()
        crop_path = os.path.join(cache_dir, f"{crop_id}.jpg")
        plt.savefig(crop_path)
        image_path_list.append(crop_path)
        # 显示可视化结果
        plt.show()
        return image_path_list
    
    def execute(self, image_path, content, box_threshold, text_threshold, save_path):
        image_source, image = load_image(image_path)
        # boxes, _, phrases = predict(model=self.model,image=image,caption=content,box_threshold=box_threshold,text_threshold=text_threshold,device=self.config["tool"]["detect"]["device"])
        h, w, _ = image_source.shape
        # torch_boxes = boxes * torch.Tensor([w, h, w, h])
        # xyxy = box_convert(boxes=torch_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        # normed_xyxy = np.around(np.clip(xyxy / np.array([w, h, w, h]), 0., 1.), 3).tolist()
        # yolo_world
        yolo_result = self.yolo_model.predict(image_path, save=True)
        # bboxes = normed_xyxy
        # phrases = phrases
        bboxes, phrases = [], []
        for result in yolo_result:
            boxes = result.boxes
            box = [[round(coord, 3) for coord in box] for box in boxes.xyxy.cpu().numpy().tolist()]
            if len(box) == 0:
                continue
            normed_box = np.around(np.clip(np.array(box) / np.array([w, h, w, h]), 0., 1.), 3).tolist()
            # normed_xyxy.extend(normed_box)
            classes = boxes.cls.cpu().numpy().tolist()
            classes = [result.names[cls] for cls in classes]
            # for cls in classes:
            #     phrases.append(result.names[cls])
            # print(normed_xyxy, phrases, normed_box, classes)
            # bboxes, phrases = self.box_Union(normed_xyxy, normed_box, phrases, classes)
            bboxes, phrases = normed_box, classes
        result = {"boxes":bboxes, "phrases":phrases, "save_path":[]}
        
        image_path_list = self.draw(bboxes, phrases, image_source, image_path, save_path)
        result["save_path"] = image_path_list
        print(result)
        return result

    # def execute(self, image_path, content, box_threshold, text_threshold, save_path):
    #     image_source, image = load_image(image_path)
    #     boxes, _, phrases = predict(model=self.model,image=image,caption=content,box_threshold=box_threshold,text_threshold=text_threshold,device=self.config["tool"]["detect"]["device"])
    #     h, w, _ = image_source.shape
    #     torch_boxes = boxes * torch.Tensor([w, h, w, h])
    #     xyxy = box_convert(boxes=torch_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    #     normed_xyxy = np.around(np.clip(xyxy / np.array([w, h, w, h]), 0., 1.), 3).tolist()
    #     # yolo_world
    #     yolo_result = self.yolo_model.predict(image_path, save=True)
    #     bboxes = normed_xyxy
    #     phrases = phrases
    #     for result in yolo_result:
    #         boxes = result.boxes
    #         box = [[round(coord, 3) for coord in box] for box in boxes.xyxy.cpu().numpy().tolist()]
    #         if len(box) == 0:
    #             continue
    #         normed_box = np.around(np.clip(np.array(box) / np.array([w, h, w, h]), 0., 1.), 3).tolist()
    #         # normed_xyxy.extend(normed_box)
    #         classes = boxes.cls.cpu().numpy().tolist()
    #         classes = [result.names[cls] for cls in classes]
    #         # for cls in classes:
    #         #     phrases.append(result.names[cls])
    #         print(normed_xyxy, phrases, normed_box, classes)
    #         bboxes, phrases = self.box_Union(normed_xyxy, normed_box, phrases, classes)
    #     result = {"boxes":bboxes, "phrases":phrases, "save_path":[]}
        
    #     image_path_list = self.draw(bboxes, phrases, image_source, image_path, save_path)
    #     result["save_path"] = image_path_list
    #     print(result)
    #     return result

import os
import argparse

from ultralytics import YOLO, RTDETR

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str)
    parser.add_argument('-weight', type=str)

    args = parser.parse_args()
    return args

def yolo_detect(weight_path, private_test_path):
    model = YOLO(weight_path)
    model.predict(source=private_test_path, show=False, save_conf=True, save=False, save_txt=True, conf=0.1, iou=0.1)

def detr_detect(weight_path, private_test_path):
    model = RTDETR(weight_path)
    model.predict(source=private_test_path, show=False, save_conf=True, save=False, save_txt=True, conf=0.2, iou=0.1)    

def main(args):
    private_test_path = 'private_test'

    model = args.model
    weight = args.weight

    if model == 'YOLO':
        yolo_detect(weight, private_test_path)
    
    if model == 'DETR':
        detr_detect(weight, private_test_path)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
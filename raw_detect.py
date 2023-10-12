import os
import shutil

def main():
    yolo_weight_paths = [
        'weights/yolo1.pt',
        'weights/yolo2.pt',
        'weights/yolo_rowcol.pt',
        'weights/yolo3.pt'
    ]

    detr_weight_path = 'weights/transfomer.pt'

    for weight_path in yolo_weight_paths:
        os.system(f'python raw_detect_run.py -model YOLO -weight {weight_path}')

    os.system(f'python raw_detect_run.py -model DETR -weight {detr_weight_path}')
    
if __name__ == '__main__':
    if os.path.exists('runs'):
        shutil.rmtree('runs', ignore_errors=True)
    main()
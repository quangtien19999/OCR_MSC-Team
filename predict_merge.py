import os

import torch.backends.cudnn as cudnn
from merge_modules import MergeModel_V2

import torch
import cv2
import numpy as np
import shutil

def preprocess_image(image_path, scale=0.3, min_width=40):
    image = cv2.imread(image_path)[..., [2,1,0]]
    image = image.transpose((2, 0, 1))
    

    c, h, w = image.shape 
    new_h = int(scale * h) if int(scale * h) > min_width else min_width
    new_w = int(scale * w) if int(scale * w) > min_width else min_width

    image = np.array([cv2.resize(image[i], (new_w, new_h), interpolation=cv2.INTER_AREA) for i in range(c)])
    img_array = np.array(image) / 255.
    img_tensor = torch.from_numpy(img_array).type(torch.FloatTensor)
    return img_tensor 


def trace_horizontal(cells, R, cur_cell, visited_matrix):
    y, x = cur_cell 
    visited_matrix[y, x] = 1
    if x < R.shape[1] and R[y, x] == 1:
        cells.append([y, x+1])
        cells = trace_horizontal(cells, R, [y, x+1], visited_matrix)
    return cells

def trace_vertical(cells, D, cur_cell, visited_matrix):
    y, x = cur_cell 
    visited_matrix[y, x] = 1 
    if y < D.shape[0] and D[y, x] == 1 and ((x == 0) or [y+1, x-1] in cells):
        cells.append([y+1, x])
        cells = trace_vertical(cells, D, [y+1, x], visited_matrix)
    
    return cells 


def predict_final_result(image_path, label_path):
    image = cv2.imread(image_path)
    h, w, c = image.shape

    data = open(label_path, 'r').read().splitlines() 
    data = [x for x in data if not x.startswith('2')]
    row_data = [] 
    col_data = []
    row_coords = [] 
    col_coords = []  
    for line in data:
        cls, x_center, y_center, width, height, score = [float(x) for x in line.split(' ')]
        if cls == 0:
            row_coords.extend([y_center - height / 2, y_center + height / 2])
        elif cls == 1:
            col_coords.extend([x_center - width / 2, x_center + width / 2])

    temp_sort = [] 
    for i in range(0, len(row_coords), 2):
        temp_sort.extend([(row_coords[i] + row_coords[i+1]) / 2 * h, (row_coords[i] + row_coords[i+1]) / 2 * h + 1e-8])
    
    row_coords = [row_coords[i] for i in np.argsort(np.array(temp_sort, dtype=np.float64))]

    temp_sort = [] 
    for i in range(0, len(col_coords), 2):
        temp_sort.extend([(col_coords[i] + col_coords[i+1]) / 2 * w, (col_coords[i] + col_coords[i+1]) / 2 * w + 1e-8])
    col_coords = [col_coords[i] for i in np.argsort(np.array(temp_sort, dtype=np.float64))]

    # row_coords.sort()
    row_coords[0] = 0
    row_coords[-1] = 1
    # col_coords.sort() 
    col_coords[0] = 0 
    col_coords[-1] = 1

    row_data = [(row_coords[i] + row_coords[i+1]) / 2 for i in range(1, len(row_coords)-1, 2)]
    col_data = [(col_coords[i] + col_coords[i+1]) / 2 for i in range(1, len(col_coords)-1, 2)]

    input_img = preprocess_image(image_path).unsqueeze(0).cuda()
    arc = row_data, col_data
    arc_c = [[torch.Tensor([y]) for y in x] for x in arc]
    pred = net(input_img,arc_c)

    u,d,l,r = pred # up, down, left, right
    # calculate D and R matrice, 
    D = 0.5 * u[:, :-1, :] * d[:, 1:, :] + 0.25 * (u[:, :-1, :] + d[:, 1:, :])
    R = 0.5 * r[:, :, :-1] * l[:, :, 1:] + 0.25 * (r[:, :, :-1] + l[:, :, 1:])
    D = D[0].detach().cpu().numpy()
    R = R[0].detach().cpu().numpy()

   

    # 0.8
    thresh = 0.5
    D[D>thresh] = 1
    D[D<=thresh] = 0
    threshR = 0.3
    R[R>threshR] = 1
    R[R<=threshR] = 0

  
    list_cells = [] 
    for i in range(0, len(row_coords), 2):
        for j in range(0, len(col_coords), 2):
            list_cells.append([col_coords[j], row_coords[i], col_coords[j+1], row_coords[i+1]])

    
    list_cells = np.array(list_cells, dtype=np.float32).reshape((len(row_coords)//2, len(col_coords)//2, -1))
    list_final_cell_ids = [] 
    visited_matrix = np.zeros((list_cells.shape[0], list_cells.shape[1]), dtype=np.int16)
    for i in range(list_cells.shape[0]):
        for j in range(list_cells.shape[1]):
            if visited_matrix[i, j] == 1:
                continue
            cell_ids = [[i, j]] 
            if i < R.shape[0] and j < R.shape[1] and R[i, j] == 1: # trace horizontal lines 
                cell_ids = trace_horizontal(cell_ids, R, [i, j], visited_matrix)
            for cell in cell_ids:
                y, x = cell 
                if y < D.shape[0] and x < D.shape[1] and D[y, x] == 1: # trace vertical 
                    cell_ids = trace_vertical(cell_ids, D, [y, x], visited_matrix)
            list_final_cell_ids.append(cell_ids)
    
    folder_save = 'merge_span_labels'
    
    with open('{}/{}.txt'.format(folder_save, os.path.basename(image_path).split('.')[0]), 'a') as f:
        pass
    for cell_ids in list_final_cell_ids: 
        cells = list_cells[tuple(zip(*cell_ids))]
        if len(cells) == 1:
            xmin, ymin, xmax, ymax = int(cells[0, 0] * image.shape[1]), int(cells[0, 1] * image.shape[0]), int(cells[0, 2] * image.shape[1]), int(cells[0, 3] * image.shape[0]) 
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
        else:
            xmin = int(np.min(cells[:, 0]) * image.shape[1])
            ymin = int(np.min(cells[:, 1]) * image.shape[0])
            xmax = int(np.max(cells[:, 2]) * image.shape[1]) - 1
            ymax = int(np.max(cells[:, 3]) * image.shape[0]) - 1  
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            
            cell_ids = np.array(cell_ids)
            start_row, end_row = np.min(cell_ids[:, 0]), np.max(cell_ids[:, 0])
            start_col, end_col = np.min(cell_ids[:, 1]), np.max(cell_ids[:, 1]) 
            
            with open('{}/{}.txt'.format(folder_save, os.path.basename(image_path).split('.')[0]), 'a') as f:
                f.write(f'{xmin} {ymin} {xmax} {ymax} {start_row} {end_row} {start_col} {end_col}\n')                       

net = MergeModel_V2(3).cuda()
cudnn.benchmark = True
cudnn.deterministic = True

net.load_state_dict(torch.load('weights/CP94_0.992126.pth'))
net.eval()

folder_image_path = 'private_test'
folder_label_path = 'labels'
list_image_name = os.listdir(folder_image_path)
list_image_name.sort()

folder_save = 'merge_span_labels'
if os.path.exists(folder_save):
    shutil.rmtree(folder_save, ignore_errors=True)

os.mkdir(folder_save)

for image_name in list_image_name:
    print(image_name)
    
    label_name = image_name.split('.')[0] + '.txt'

    predict_final_result(os.path.join(folder_image_path, image_name), os.path.join(folder_label_path, label_name))


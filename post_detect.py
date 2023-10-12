import numpy as np
import os
import cv2
import csv
import shutil

def nms(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

        inds = np.where(ovr < thresh)[0]
        # outds = np.where(ovr > 0)[0]
        # if len(outds) < 3:
        keep.append(i)
        order = order[inds + 1]

    return keep

def nms_special_row(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    lines = (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = h / lines[i] 

        inds = np.where(ovr < thresh)[0]
        # outds = np.where(ovr > 0)[0]
        # if len(outds) < 3:
        keep.append(i)
        order = order[inds + 1]
    
    return keep

def nms_special_col(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    lines = (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w / lines[i] 

        inds = np.where(ovr < thresh)[0]
        # outds = np.where(ovr > 0)[0]
        # if len(outds) < 3:
        keep.append(i)
        order = order[inds + 1]
    
    return keep
    

def nms_span(dets, thresh):
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])

        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

        inds = np.where(ovr < thresh)[0]
        order = order[inds + 1]

        outds = np.where(ovr > thresh)[0]
        if len(outds) > 0:
            keep.append(i)
    return keep


def find_vertical_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = ~cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    kernel = np.ones((1, 5), np.uint8)  # Kernel for dilation
    erode_edges = cv2.erode(vertical_mask.copy(), kernel, iterations=1)
    dilated_edges = cv2.dilate(erode_edges.copy(), kernel, iterations=1)
    vertical_mask = vertical_mask - dilated_edges
    vertical_mask = cv2.dilate(vertical_mask.copy(), kernel, iterations=1)

    return vertical_mask

def find_edge(image, x_offset):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(2*image.shape[0]/3))

    lines_point = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Filter lines based on their orientation (vertical lines have theta close to 0 or pi)
            if (theta < np.pi/8 or theta > 7*np.pi/8) and (theta < 9*np.pi/8 or theta > 13*np.pi/8):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                lines_point.append(x1 + x_offset)
                lines_point.append(x2 + x_offset)
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    return lines_point

def span_merge(span, img, c_info, col):
    st_c, e_c = c_info
    vertical_mask = find_vertical_mask(img)
    x_min, y_min, x_max, y_max = span 

    sub_vertical_mask = vertical_mask[y_min: y_max, x_min:x_max]

    line_points = find_edge(np.stack((sub_vertical_mask, sub_vertical_mask, sub_vertical_mask), axis=-1), x_offset=x_min)

    need_split_col_list = [] 
    for index in range(st_c, e_c):
        s = col[index*2+1]
        e = col[(index+1)*2] 

        ns = s - abs(e - s) * 0.1 
        ne = e + abs(e - s) * 0.1 
        flag_split = False 
        
        for point in line_points:
            if point >= ns and point <= ne: 
                flag_split = True 
                break 
        if flag_split:
            need_split_col_list.append(index) 
    return need_split_col_list

def check_special_condition(span_poss, row, col, remove_id):
    nc = len(col) // 2 
    nr = len(row) // 2 
    outlier = [x for x in span_poss if x[1][0] != x[1][1] and x[1][0] >= nr-4]
    if len(row) // 2 <= 4 or len(outlier) > 0:
        return span_poss, remove_id 

    row_4_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr - 1]
    row_3_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr-2]
    row_2_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr-3]
    row_1_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr-4]

    if len(row_4_span) == 1 and len(row_3_span) == 1  and len(row_2_span) == 1 and len(row_1_span) == 1 :
        # fix 4th 
        if row_4_span[0][1][2] == 0 and row_4_span[0][1][3] == nc - 1:
            pass 
        else: 
            row_4_span[0][1][2] = 0 
            row_4_span[0][1][3] = nc-1
            row_4_span[0][0][0] = 0 
            row_4_span[0][0][2] = col[-1] 

        # fix 3th 
        if row_3_span[0][1][2] == 0 and row_3_span[0][1][3] == nc-2:
            pass
        else:
            row_3_span[0][1][2] = 0 
            row_3_span[0][1][3] = nc-2 
            row_3_span[0][0][0] = 0 
            row_3_span[0][0][2] = col[-3]
        # fix 2nd 
        if row_2_span[0][1][2] == 0 and row_2_span[0][1][3] == nc-2:
            pass
        else:
            row_2_span[0][1][2] = 0 
            row_2_span[0][1][3] = nc-2
            row_2_span[0][0][0] = 0 
            row_2_span[0][0][2] = col[-3]
        # fix 1
        if row_1_span[0][1][2] == 0 and row_1_span[0][1][3] == nc-2:
            pass
        else:
            row_1_span[0][1][2] = 0 
            row_1_span[0][1][3] = nc-2
            row_1_span[0][0][0] = 0 
            row_1_span[0][0][2] = col[-3]
        
        for y in range(nr-4, nr):
            for x in range(nc):
                id = y*nc + x
                if x == nc-1:
                    if y != nr-1 and id in remove_id:
                        remove_id.remove(id)
                else:
                    remove_id.append(id) 
        return span_poss, remove_id

    return span_poss, remove_id 
    

def res_converter(private_test, yolo_path, yolo_path_v7, trans_path, row_col_path, yolo_path_v9):
    files = os.listdir(os.path.join(yolo_path, 'labels'))
    results = []

    for file in files:
        print(file)
        yolo_file = os.path.join(yolo_path, 'labels', file)
        yolo_file_v7 = os.path.join(yolo_path_v7, 'labels', file)
        trans_file = os.path.join(trans_path, 'labels', file)
        row_col_file = os.path.join(row_col_path, 'labels', file)
        yolo_file_v9 = os.path.join(yolo_path_v9, 'labels', file)

        img_file = os.path.join(private_test, file.split('.')[0] + '.jpg')

        img = cv2.imread(img_file)
        img_H, img_W = img.shape[:2]
        
        row = []
        col = []
        span = []

        with open(yolo_file, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height, score = [float(x) for x in line.split(' ')]

                x_min = int((x_center - width / 2) * img_W)
                x_max = int((x_center + width / 2) * img_W)
                y_min = int((y_center - height / 2) * img_H)
                y_max = int((y_center + height / 2) * img_H)

                if class_id == 0 and score > 0.2:
                    row.append([0, int(y_min), img_W, int(y_max), score])
                elif class_id == 1 and score > 0.2:
                    col.append([int(x_min), 0, int(x_max), img_H, score])
                else:
                    if score > 0.2:
                # if class_id == 2 and score > 0.2:
                        span.append([int(x_min), int(y_min), int(x_max), int(y_max), score])
        f.close()

        with open(trans_file, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height, score = [float(x) for x in line.split(' ')]

                x_min = int((x_center - width / 2) * img_W)
                x_max = int((x_center + width / 2) * img_W)
                y_min = int((y_center - height / 2) * img_H)
                y_max = int((y_center + height / 2) * img_H)

                if class_id == 2:
                    span.append([int(x_min), int(y_min), int(x_max), int(y_max), score])
        f.close()

        with open(yolo_file_v7, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height, score = [float(x) for x in line.split(' ')]

                x_min = int((x_center - width / 2) * img_W)
                x_max = int((x_center + width / 2) * img_W)
                y_min = int((y_center - height / 2) * img_H)
                y_max = int((y_center + height / 2) * img_H)
                if class_id == 2 and score > 0.2:
                    span.append([int(x_min), int(y_min), int(x_max), int(y_max), score])
        f.close()

        with open(row_col_file, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height, score = [float(x) for x in line.split(' ')]

                x_min = int((x_center - width / 2) * img_W)
                x_max = int((x_center + width / 2) * img_W)
                y_min = int((y_center - height / 2) * img_H)
                y_max = int((y_center + height / 2) * img_H)
                if class_id == 0:
                    row.append([0, int(y_min), img_W, int(y_max), score])
        f.close()

        with open(yolo_file_v9, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height, score = [float(x) for x in line.split(' ')]

                x_min = int((x_center - width / 2) * img_W)
                x_max = int((x_center + width / 2) * img_W)
                y_min = int((y_center - height / 2) * img_H)
                y_max = int((y_center + height / 2) * img_H)
                if class_id == 1:
                    col.append([int(x_min), 0, int(x_max), img_H, score])
                elif class_id == 2 and score > 0.2:
                    span.append([int(x_min), int(y_min), int(x_max), int(y_max), score])

        f.close()

        row = np.array(row)
        col = np.array(col)

        keep_row = nms(row, 0.05)
        keep_col = nms(col, 0.1)

        row = row[keep_row, :]
        keep_row_special = nms_special_row(row, 0.8)
        row = row[keep_row_special, :]

        col = col[keep_col, :]

        for r in row:
            center_x = (r[0] + r[2]) / 2 / img_W
            center_y = (r[1] + r[3]) / 2 / img_H
            width = (r[2] - r[0]) / img_W 
            height = (r[3] - r[1]) / img_H

            with open(os.path.join('labels', file), 'a') as f:
                f.write(f'0 {center_x} {center_y} {width} {height} {r[4]}\n')

        for c in col:
            center_x = (c[0] + c[2]) / 2 / img_W
            center_y = (c[1] + c[3]) / 2 / img_H
            width = (c[2] - c[0]) / img_W 
            height = (c[3] - c[1]) / img_H

            with open(os.path.join('labels', file), 'a') as f:
                f.write(f'1 {center_x} {center_y} {width} {height} {c[4]}\n')

        row = row[:, [1, 3]].astype(np.int32).flatten().tolist()
        col = col[:, [0, 2]].astype(np.int32).flatten().tolist()
        
        temp_sort = [] 
        for i in range(0, len(row), 2):
            temp_sort.extend([(row[i] + row[i+1]) / 2, (row[i] + row[i+1]) / 2 + 1e-9])
        
        row = [row[i] for i in np.argsort(temp_sort)]

        temp_sort = [] 
        for i in range(0, len(col), 2):
            temp_sort.extend([(col[i] + col[i+1]) / 2, (col[i] + col[i+1]) / 2 + 1e-9])
        col = [col[i] for i in np.argsort(temp_sort)]

        row[0] = 0
        row[-1] = img_H
        col[0] = 0 
        col[-1] = img_W

        cell = []
        for i in range(0, len(row), 2):
            for j in range(0, len(col), 2):
                r1, r2 = row[i], row[i+1]
                c1, c2 = col[j], col[j+1]
                bbox = [c1,r1,c2,r2]
                pos = [int(i/2),int(j/2)]
                cell_info = [bbox, pos]
                cell.append(cell_info)

        if len(span) > 0:
            span = np.array(span)
            keep_span = nms_span(span, 0.1)
            span = span[keep_span, :]
            span = span[:, [0, 1, 2, 3]].astype(np.int32).tolist()

            span_poss = []
            cell_rm_id = []
            for s in span:
                flag_inter_span = True
                cell_rm_id_temp = []
                span_pos = []
                for i, cel in enumerate(cell):
                    s_x_min, s_y_min, s_x_max, s_y_max = s
                    cel_x_min, cel_y_min, cel_x_max, cel_y_max = cel[0]

                    inter_x_min = max(s_x_min, cel_x_min)
                    inter_y_min = max(s_y_min, cel_y_min)
                    inter_x_max = min(s_x_max, cel_x_max)
                    inter_y_max = min(s_y_max, cel_y_max)

                    if inter_x_min > inter_x_max or inter_y_min > inter_y_max:
                        inter_area = 0
                        continue
                    else:
                        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

                        span_area = (s_x_max - s_x_min) * (s_y_max - s_y_min)
                        cel_area = (cel_x_max - cel_x_min) * (cel_y_max - cel_y_min)

                        inter_span = inter_area / (span_area + 1e-7)
                        inter_cel = inter_area / (cel_area + 1e-7)

                        if inter_span > 0.6:
                            flag_inter_span = False
                            span_pos.append(cel[1])
                            if i not in cell_rm_id_temp:
                                cell_rm_id_temp.append(i)       

                        elif inter_cel > 0.5:
                            span_pos.append(cel[1])
                            if i not in cell_rm_id_temp:
                                cell_rm_id_temp.append(i)

                if len(span_pos) == 0:
                    continue

                if len(span_pos) == 1 and not flag_inter_span:
                    continue

                span_pos = np.array(span_pos)
                span_st_r = np.min(span_pos[:,0])
                span_e_r = np.max(span_pos[:,0])
                span_st_c = np.min(span_pos[:,1])
                span_e_c = np.max(span_pos[:,1])

                span_info = [[col[span_st_c*2], row[span_st_r*2], col[span_e_c*2+1], row[span_e_r*2+1]], [span_st_r, span_e_r, span_st_c, span_e_c]]
                cell_rm_id.extend(cell_rm_id_temp)
                span_poss.append(span_info)
            
            span_poss, cell_rm_id = check_special_condition(span_poss, row, col, cell_rm_id)

            for sp in span_poss:
                need_split_col_index_list = span_merge((sp[0][0], sp[0][1], sp[0][2], sp[0][3]), img, [sp[1][2], sp[1][3]], col)
                if len(need_split_col_index_list) == 0:
                    res = [f"{file.split('.')[0]}", f"{sp[0][0]} {sp[0][1]} {sp[0][2]} {sp[0][3]} {sp[1][0]} {sp[1][1]} {sp[1][2]} {sp[1][3]}"]
                    results.append(res)
                else:
                    temp_start_col = sp[1][2]
                    for id in need_split_col_index_list: 
                        left_span_cell = [f"{file.split('.')[0]}", f"{col[temp_start_col*2]} {sp[0][1]} {col[id*2+1]} {sp[0][3]} {sp[1][0]} {sp[1][1]} {temp_start_col} {id}"]
                        results.append(left_span_cell)
                        temp_start_col = id+1

                    
                    id = sp[1][3]
                    last_left_span_cell = [f"{file.split('.')[0]}", f"{col[temp_start_col*2]} {sp[0][1]} {col[id*2+1]} {sp[0][3]} {sp[1][0]} {sp[1][1]} {temp_start_col} {id}"]
                    results.append(last_left_span_cell)

                center_x = (sp[0][0] + sp[0][2]) / 2 / img_W
                center_y = (sp[0][1] + sp[0][3]) / 2 / img_H
                width = (sp[0][2] - sp[0][0]) / img_W 
                height = (sp[0][3] - sp[0][1]) / img_H

                with open(os.path.join('labels', file), 'a') as f:
                    f.write(f'2 {center_x} {center_y} {width} {height} 0.99\n')

            sel_cell = [cell[i] for i in range(len(cell)) if i not in cell_rm_id]
        else:
            sel_cell = cell

        for sc in sel_cell:
            res = [f"{file.split('.')[0]}", f"{sc[0][0]} {sc[0][1]} {sc[0][2]} {sc[0][3]} {sc[1][0]} {sc[1][0]} {sc[1][1]} {sc[1][1]}"]
            results.append(res)


if __name__ == '__main__':
    if os.path.exists('labels'):
        shutil.rmtree('labels', ignore_errors=True)
    
    os.mkdir('labels')

    res_converter('private_test',
                  'runs/detect/predict',
                  'runs/detect/predict2',
                  'runs/detect/predict5',
                  'runs/detect/predict3',
                  'runs/detect/predict4')
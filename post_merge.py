import numpy as np
import os
import cv2
import csv

def calculate_intersection_area(box1, box2):
    # Extract coordinates for each box
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate the intersection coordinates
    x1_intersect = max(x1_1, x1_2)
    y1_intersect = max(y1_1, y1_2)
    x2_intersect = min(x2_1, x2_2)
    y2_intersect = min(y2_1, y2_2)
    
    # Check if there is a valid intersection (non-negative area)
    if x1_intersect < x2_intersect and y1_intersect < y2_intersect:
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
    else:
        intersection_area = 0.0  # No intersection
    
    return intersection_area


def calculate_iou(box1, box2):
    # Extract coordinates for each box
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate the intersection coordinates
    x1_intersect = max(x1_1, x1_2)
    y1_intersect = max(y1_1, y1_2)
    x2_intersect = min(x2_1, x2_2)
    y2_intersect = min(y2_1, y2_2)
    
    # Check if there is a valid intersection (non-negative area)
    if x1_intersect < x2_intersect and y1_intersect < y2_intersect:
        intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
    else:
        intersection_area = 0.0  # No intersection
    
    # Calculate the union area
    union_area = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - intersection_area
    
    # Calculate the IoU
    if union_area > 0:
        iou = intersection_area / union_area
    else:
        iou = 0.0
    
    return iou

def find_vertical_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = ~cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

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
                lines_point.append((x1+x2) / 2 + x_offset)
                # print(x1, y1, x2, y2)
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    return image , lines_point

def span_merge(span, img, c_info, col):
    st_c, e_c = c_info
    vertical_mask = find_vertical_mask(img)
    x_min, y_min, x_max, y_max = span 

    # cv2.imshow('origin', img[y_min: y_max, x_min:x_max])


    sub_vertical_mask = vertical_mask[y_min: y_max, x_min:x_max]
    # print(sub_vertical_mask.shape)

    edged_image, line_points = find_edge(np.stack((sub_vertical_mask, sub_vertical_mask, sub_vertical_mask), axis=-1), x_offset=x_min)
    # print("line_points", line_points)

    need_split_col_list = [] 
    for index in range(st_c, e_c):
        s = col[index*2+1]
        e = col[(index+1)*2] 

        ns = s - (e - s) * 0.1 
        ne = e + (e - s) * 0.1 
        flag_split = False 
        for point in line_points:
            if point >= ns and point <= ne: 
                flag_split = True 
                # print(point, ns, ne)
                # img = cv2.rectangle(img, (int(ns), y_min), (int(ne), y_max), (255, 255, 255), 2)
                # img = cv2.circle(img, (int(point), (y_min  + y_max) // 2), radius=3, color=(255, 0, 0), thickness=-1)
                break 
        if flag_split:
            need_split_col_list.append(index) 
    
    # print("need_split_col_list", need_split_col_list)

    
    # cv2.imshow('img', imutils.resize(img.copy(), width=720))
    # cv2.imshow('mask', sub_vertical_mask)
    # cv2.imshow('edge', edged_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return need_split_col_list

def check_special_condition(span_poss, row, col, remove_id):
    nc = len(col) // 2 
    nr = len(row) // 2 
    outlier = [x for x in span_poss if x[1][0] != x[1][1] and x[1][0] >= nr-4]
    if len(row) // 2 <= 4 or len(outlier) > 0:
        return span_poss, remove_id
    # if len(row) // 2 <= 4:
    #     return span_poss, remove_id 
    row_4_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr - 1]
    row_3_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr-2]
    row_2_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr-3]
    row_1_span = [x for id, x  in enumerate(span_poss) if x[1][0] == x[1][1] and x[1][0] == nr-4]

    if len(row_4_span) == 1 and len(row_3_span) == 1  and len(row_2_span) == 1 and len(row_1_span) == 1 :
        # fix 4th 
        # print(row_4_span)
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
    

def check_merge_span_detect(origin_span_info, merge_span_detect_txt_path, cell_rm_id_temp, rows, cols):
    merge_span_detect_data = open(merge_span_detect_txt_path, 'r').read().splitlines() 
    merge_span_detect_data = [line for line in merge_span_detect_data if line.strip() != ""]
    if len(merge_span_detect_data) == 0:
        return origin_span_info, cell_rm_id_temp, False
    o_xmin, o_ymin, o_xmax, o_ymax = origin_span_info[0]
    o_start_row, o_end_row, o_start_col, o_end_col = origin_span_info[1]
    span_valid_flag = False 
    for line in merge_span_detect_data:
        xmin, ymin, xmax, ymax, start_row, end_row, start_col, end_col = [int(x) for x in line.split(' ')]
        iou = calculate_iou([xmin, ymin, xmax, ymax], [o_xmin, o_ymin, o_xmax, o_ymax])
        flag = ((ymax - ymin) * (xmax - xmin)) < ((o_ymax - o_ymin) * (o_xmax - o_xmin))

        if iou < 0.1 or flag:
            continue
        span_valid_flag = True
        origin_span_info[0] = [xmin, ymin, xmax, ymax]
        origin_span_info[1] = [start_row, end_row, start_col, end_col]
        cell_rm_id_temp = [y * (len(cols) // 2) + x for x in range(int(start_col), int(end_col+1)) for y in range(int(start_row), int(end_row+1))]
        break 
    return origin_span_info, cell_rm_id_temp, span_valid_flag

def find_additional_merge_span_detect(detected_span, merge_span_detect_txt_path, cols):
    merge_span_detect_data = open(merge_span_detect_txt_path, 'r').read().splitlines() 
    merge_span_detect_data = [line for line in merge_span_detect_data if line.strip() != ""]
    if len(merge_span_detect_data) == 0:
        return None, None 
    
    additional_merge_span = [] 
    additional_cell_rm_id = [] 
    for line in merge_span_detect_data:
        xmin, ymin, xmax, ymax, start_row, end_row, start_col, end_col = [int(x) for x in line.split(' ')] 
        used_flag = True
        for origin_span_info in detected_span:
            o_xmin, o_ymin, o_xmax, o_ymax = origin_span_info[0]
            iou = calculate_iou([xmin, ymin, xmax, ymax], [o_xmin, o_ymin, o_xmax, o_ymax])
            if iou > 0.1: 
                used_flag = False 
                break 
        
        if used_flag: 
            additional_merge_span.append([[xmin, ymin, xmax, ymax], [start_row, end_row, start_col, end_col]])
            additional_cell_rm_id.extend([y * (len(cols) // 2) + x for x in range(int(start_col), int(end_col+1)) for y in range(int(start_row), int(end_row+1))])
    
    return additional_merge_span, additional_cell_rm_id
            

def res_converter(data_path):
    files = os.listdir(os.path.join(data_path))
    files.sort()
    results = []

    for file in files:
        results_json = []
        print(file)
        res_file = os.path.join(data_path, file)
        # img_file = os.path.join(data_path, file.split('.')[0] + '.jpg')
        img_file = os.path.join('private_test', file.split('.')[0] + '.jpg')
        if not os.path.exists(img_file):
            continue

        img = cv2.imread(img_file)
        img_H, img_W = img.shape[:2]
        
        row = []
        col = []
        span = []

        with open(res_file, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height, score = [float(x) for x in line.split(' ')]

                x_min = int((x_center - width / 2) * img_W)
                x_max = int((x_center + width / 2) * img_W)
                y_min = int((y_center - height / 2) * img_H)
                y_max = int((y_center + height / 2) * img_H)

                if class_id == 0:
                    row.extend([int(y_min), int(y_max)])
                elif class_id == 1:
                    col.extend([int(x_min), int(x_max)])
                else:
                    span.append([int(x_min), int(y_min), int(x_max), int(y_max)])

        # Sort row and col for correct table idx
        temp_sort = [] 
        for i in range(0, len(row), 2):
            temp_sort.extend([(row[i] + row[i+1]) / 2, (row[i] + row[i+1]) / 2 + 1])
        
        # print(temp_sort)
        row = [row[i] for i in np.argsort(temp_sort)]

        temp_sort = [] 
        for i in range(0, len(col), 2):
            temp_sort.extend([(col[i] + col[i+1]) / 2, (col[i] + col[i+1]) / 2 + 1])
        col = [col[i] for i in np.argsort(temp_sort)]


        row[0] = 0 
        row[-1] = img_H-1
        col[0] = 0 
        col[-1] = img_W-1
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
            span_poss = []
            cell_rm_id = []
            for s in span:
                span_pos = []
                s_x_min, s_y_min, s_x_max, s_y_max = s
                flag_inter_span = True
                cell_rm_id_temp = []
                for i, cel in enumerate(cell):
                    cel_x_min, cel_y_min, cel_x_max, cel_y_max = cel[0]

                    inter_x_min = max(s_x_min, cel_x_min)
                    inter_y_min = max(s_y_min, cel_y_min)
                    inter_x_max = min(s_x_max, cel_x_max)
                    inter_y_max = min(s_y_max, cel_y_max)

                    if inter_x_min > inter_x_max or inter_y_min > inter_y_max:
                        inter_area = 0
                        # continue
                    else:
                        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

                    span_area = (s_x_max - s_x_min) * (s_y_max - s_y_min)
                    cel_area = (cel_x_max - cel_x_min) * (cel_y_max - cel_y_min)

                    inter_span = inter_area / (span_area + 1e-7)
                    inter_cel = inter_area / (cel_area + 1e-7)
                  
                    if inter_span > 0.6:
                        # break
                        flag_inter_span = False
                        span_pos.append(cel[1])
                        if i not in cell_rm_id_temp:
                            cell_rm_id_temp.append(i)       

                    elif inter_cel > 0.5:
                        span_pos.append(cel[1])
                        if i not in cell_rm_id_temp:
                            cell_rm_id_temp.append(i)

                if len(span_pos) == 0:
                    print('SPAN POS 0')
                    continue

                if len(span_pos) == 1 and not flag_inter_span:
                    print('SPAN POS 1')

                    continue
                    
                span_pos = np.array(span_pos)
                span_st_r = np.min(span_pos[:,0])
                span_e_r = np.max(span_pos[:,0])
                span_st_c = np.min(span_pos[:,1])
                span_e_c = np.max(span_pos[:,1])
                span_info = [[col[span_st_c*2], row[span_st_r*2], col[span_e_c*2+1], row[span_e_r*2+1]], [span_st_r, span_e_r, span_st_c, span_e_c]]

                span_info, cell_rm_id_temp, span_valid_flag = check_merge_span_detect(span_info, 
                                                                     os.path.join('merge_span_labels', file),
                                                                     cell_rm_id_temp, row, col)
                # if span_valid_flag:
                cell_rm_id.extend(cell_rm_id_temp)
                span_poss.append(span_info)
            
         
            span_poss, cell_rm_id = check_special_condition(span_poss, row, col, cell_rm_id)

            for sp in span_poss:
                need_split_col_index_list = span_merge((sp[0][0], sp[0][1], sp[0][2], sp[0][3]), img, [sp[1][2], sp[1][3]], col)

                # print(need_split_col_index_list)
                if len(need_split_col_index_list) == 0:
                    res = [f"{file.split('.')[0]}", f"{sp[0][0]} {sp[0][1]} {sp[0][2]} {sp[0][3]} {sp[1][0]} {sp[1][1]} {sp[1][2]} {sp[1][3]}"]
                    results.append(res)
                    img = cv2.rectangle(img, (sp[0][0], sp[0][1]), (sp[0][2], sp[0][3]), (0,255,0), 3)
                    img = cv2.putText(img, f'{sp[1][0]}, {sp[1][1]}, {sp[1][2]}, {sp[1][3]}', (sp[0][0], sp[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                else:
                    temp_start_col = sp[1][2]
                    for id in need_split_col_index_list: 
                        left_span_cell = [f"{file.split('.')[0]}", f"{col[temp_start_col*2]} {sp[0][1]} {col[id*2+1]} {sp[0][3]} {sp[1][0]} {sp[1][1]} {temp_start_col} {id}"]
                       
                        results.append(left_span_cell)
                        img = cv2.rectangle(img, (col[temp_start_col*2], sp[0][1]), (col[id*2+1], sp[0][3]), (0,255,0), 3)
                        img = cv2.putText(img, f'{sp[1][0]}, {sp[1][1]}, {temp_start_col}, {id}', (col[temp_start_col*2], sp[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        img = cv2.putText(img, f'span', (col[temp_start_col*2], sp[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        temp_start_col = id+1

                    
                    id = sp[1][3]
                    last_left_span_cell = [f"{file.split('.')[0]}", f"{col[temp_start_col*2]} {sp[0][1]} {col[id*2+1]} {sp[0][3]} {sp[1][0]} {sp[1][1]} {temp_start_col} {id}"]
                 
                    results.append(last_left_span_cell)

                    img = cv2.rectangle(img, (col[temp_start_col*2], sp[0][1]), (col[id*2+1], sp[0][3]), (0,255,0), 3)
                    img = cv2.putText(img, f'{sp[1][0]}, {sp[1][1]}, {temp_start_col}, {id}', (col[temp_start_col*2], sp[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    img = cv2.putText(img, f'span', (col[temp_start_col*2], sp[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            cell_rm_id = list(set(cell_rm_id))
            sel_cell = [cell[i] for i in range(len(cell)) if i not in cell_rm_id]
        else:
            sel_cell = cell

        for sc in sel_cell:
            res = [f"{file.split('.')[0]}", f"{sc[0][0]} {sc[0][1]} {sc[0][2]} {sc[0][3]} {sc[1][0]} {sc[1][0]} {sc[1][1]} {sc[1][1]}"]
            results.append(res)

            img = cv2.rectangle(img, (sc[0][0], sc[0][1]), (sc[0][2], sc[0][3]), (255,0,0), 3)
            img = cv2.putText(img, f'{sc[1][0]}, {sc[1][0]}, {sc[1][1]}, {sc[1][1]}', (sc[0][0], sc[0][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if not os.path.isdir('visualize'):
            os.mkdir('visualize')
        cv2.imwrite(os.path.join('visualize', file.split('.')[0] + '.jpg'), img)

    # Export to csv
    header = ['id','prediction']   
    with open('prediction.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(header)
        write.writerows(results)    

if __name__ == '__main__':
    res_converter('labels')
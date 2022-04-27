import numpy as np
import cv2 as cv
from imageio import imread
import matplotlib.pyplot as plt

'''
+ Нахождение многоугольника
+ Нахождение предметов
+ Идентификация предметов
+ Размещение предметов в многоугольнике 
'''

#  Индекс соответствует номеру фотографии в папке Items
items_list = ['hemisphere', 'marker', 'corrector', 'varnish', 'box',
              'brush', 'notepad', 'key', 'fox', 'penguin', 'not identified']


# Наложение фильтров для выделения листа бумаги и объектов
def apply_filters(img, min_color, max_color):
    bin_img = cv.inRange(img, np.array(min_color), np.array(max_color))  # пороговая фильтрация
    st = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))  # устранение черного шума
    filter = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, st)
    st = cv.getStructuringElement(cv.MORPH_RECT, (20, 20), (-1, -1))  # устранение белого шума
    edges_filters_img = cv.morphologyEx(filter, cv.MORPH_OPEN, st)
    return edges_filters_img


# Нахождение контура лежащего на листе объекта среди всех найденных контуров
def find_on_paper_contour(all_contours, hierarchy):
    max_area = 0
    paper_ind = 0
    # у листа площадь внутри контура будет наибольшей
    for i in range(len(all_contours)):
        if (cv.contourArea(all_contours[i])) > max_area:
            max_area = cv.contourArea(all_contours[i])
            paper_ind = i
    # контур объекта (многоугольника) вкладывается в контур листа
    # у многоугольника берется внешний контур
    polygon_contour = all_contours[hierarchy[0, paper_ind, 3]]
    return polygon_contour


# Нахождение контура многоугольника или любого другого объекта, лежащего на листе
def find_object_on_paper(img, min_color):
    max_color = [255, 255, 255]
    edges_filters_img = apply_filters(img, min_color, max_color)
    all_contours, hierarchy = cv.findContours(edges_filters_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    polygon_contour = find_on_paper_contour(all_contours, hierarchy)
    cv.drawContours(img, [polygon_contour], -1, [0, 0, 255], 3)
    return img, polygon_contour


# Нахождение контура многоугольника
def find_polygon(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    min_color = [0, 170, 170]
    img, polygon_contour = find_object_on_paper(img, min_color)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img, polygon_contour


# Нахождение контуров предметов среди всех найденных контуров
def find_objects_contours(all_contours):
    #  у листа площадь внутри контура будет наибольшей
    max_area = cv.contourArea(max(all_contours, key=cv.contourArea))
    min_area = max_area / 200
    objects_contours = []
    for contour in all_contours:
        area = cv.contourArea(contour)
        if min_area < area < max_area:
            objects_contours.append(contour)
    return objects_contours


#  Поиск координат объектов по переданным контурам
def find_coordinates_objects(contours, img):
    objects_coord = {}
    img_objects = {}
    for i in range(0, len(contours)):
        # определение геометрии прямоугольника, описывающего предмет(его контур)
        ((x_center, y_center), (width, height), angle) = cv.minAreaRect(contours[i])
        x_center, y_center, width, height, angle = int(x_center), int(y_center), int(width), int(height), angle
        # поворот прямоугольника, чтобы его можно было вырезать из общего изображения
        rotation_matrix = cv.getRotationMatrix2D((x_center, y_center), angle, 1)
        img_rotate = cv.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        x, y = x_center - int(width / 2), y_center - int(height / 2)
        img_obj = img_rotate[y - 17:y + height + 17, x - 17:x + width + 17]
        img_objects[i] = img_obj
        # некоторые геометрические соотношения
        objects_coord[i] = (min(width, height) / max(width, height), cv.contourArea(contours[i]) / (width * height))
    return objects_coord, img_objects


# Поиск объектов на тестовом изображении
def find_objects(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    min_color = [40, 40, 60]
    max_color = [100, 100, 255]
    edges_filters_img = apply_filters(img_hsv, min_color, max_color)
    edges_filters_img = np.invert(edges_filters_img)  # белое на черном, чтобы граница всего фото не выделялась
    all_contours, hierarchy = cv.findContours(edges_filters_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    objects_contours = find_objects_contours(all_contours)
    objects_coord, img_objects = find_coordinates_objects(objects_contours, img)
    img_with_contours = img.copy()
    cv.drawContours(img_with_contours, objects_contours, -1, [0, 0, 255], 3)
    return img_with_contours, objects_coord, img_objects, objects_contours


# Поиск объектов на эталонном изображении (на фоне листа)
def get_reference_objects():
    objects_coord = {}
    img_objects = {}
    objects_contours = {}
    for i in range(0, 10):
        img = imread('Items/item_' + str(i) + '.jpg')
        min_color = [190, 190, 0]
        img_with_contour, object_contour = find_object_on_paper(img.copy(), min_color)
        obj, img_obj = find_coordinates_objects([object_contour], img)
        img_objects[i] = img_obj[0]
        objects_coord[i] = obj[0]
        objects_contours[i] = object_contour
    return objects_coord, img_objects, objects_contours


# Сравнение изображений найденных объектов с эталонными по особым точкам
def search_matches_points(img_find_obj, img_ref_objects):
    correct_matches_dct = {}
    for j in range(len(img_ref_objects)):
        orb = cv.ORB_create()
        # особые точки и дескрипторы
        kp1, des1 = orb.detectAndCompute(img_find_obj, None)
        kp2, des2 = orb.detectAndCompute(img_ref_objects[j], None)
        # поиск двух лучших совпадений
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        correct_matches = []
        for m, n in matches:
            # совпадение
            if m.distance < 0.85 * n.distance:
                correct_matches.append([m])
        correct_matches_dct[items_list[j]] = len(correct_matches)
    # выбираем несколько лучших совпадений
    correct_matches_dct = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    return list(correct_matches_dct.keys())[0:5]


# Сравнение изображений найденных объектов с эталонными по геометрическим признакам
def search_matches_geometry(find_obj_coord, ref_objects_coord):
    similar_items = []
    for i in range(len(ref_objects_coord)):
        wh = ref_objects_coord[i][0]  # отношение ширины объекта к высоте
        area = ref_objects_coord[i][1]  # отношение площади объекта к площади ограничивающего прямоугольника
        if wh - 0.1 < find_obj_coord[0] < wh + 0.1 and area - 0.1 < find_obj_coord[1] < area + 0.1:
            similar_items.append(items_list[i])
    return similar_items


# Идентификация найденных объектов
def indentify_objects(img_find_objects, img_ref_objects, find_objects_coords, ref_objects_coord):
    items = []
    for i in range(len(img_find_objects)):
        # проверка на зеленую полусферу (плохо определяется только она) по геомерии и цвету
        if 0.9 < find_objects_coords[i][0] < 1.1:
            (x, y) = (int(len(img_find_objects[i]) / 2)), int(len(img_find_objects[i][0]) / 2)  #
            [a, b, c] = img_find_objects[i][x, y]
            if 150 < a < 190 and 170 < b < 210 and 40 < c < 55:
                items.append(items_list[0])
                continue
        # проверка по двум признакам: геометрическим и особым точкам
        similar_items_geometry = search_matches_geometry(find_objects_coords[i], ref_objects_coord)
        similar_items_points = search_matches_points(img_find_objects[i], img_ref_objects)
        item = [x for x in similar_items_points if x in set(similar_items_geometry)]
        if len(item) == 0:
            items.append(similar_items_geometry[0])
        else:
            items.append(item[0])
    return items


# Сдвиг контура по осям на величины step_x, step_y
def move_contour(contour, step_x, step_y):
    return contour + [int(step_x), int(step_y)]


# Поворот контура на угол angle (в радианах) вокруг центра
def rotate_contour(contour, angle):
    M = cv.moments(contour)
    if M['m00'] != 0:
        # координаты центра контура
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    new_contour = move_contour(contour, -cx, -cy)
    ox, oy = new_contour[:, 0, 0], new_contour[:, 0, 1]
    new_x = np.hypot(ox, oy) * np.cos(np.arctan2(oy, ox) + angle)
    new_y = np.hypot(ox, oy) * np.sin(np.arctan2(oy, ox) + angle)
    new_contour[:, 0, 0] = new_x
    new_contour[:, 0, 1] = new_y
    new_contour = move_contour(new_contour, cx, cy)
    return new_contour


#  Проверка на то, поместился ли предмет в многоугольник
def if_place(contour, img_placement):
    img_contour = img_placement.copy()
    cv.fillPoly(img_contour, [contour], (255, 255, 255))
    return np.all(np.logical_xor(np.logical_not(img_contour), img_placement))


# Попытка размещения в многоугольнике одного предмета с контуром contour
def try_place(x_start, y_start, x_finish, y_finish, img_placement, contour, step, step_angle):
    for y_place in range(y_start, y_finish, step):
        for x_place in range(x_start, x_finish, step):
            if if_place(contour, img_placement):
                return True, contour
            angle = step_angle
            while angle < 2 * np.pi:
                rot_contour = rotate_contour(contour, angle)
                angle += step_angle
                if if_place(rot_contour, img_placement):
                    return True, rot_contour
            contour = move_contour(contour, step, 0)
        contour = move_contour(contour, -x_finish + x_start, step)
    return False, contour


# Размещение предметов в многоугольнике
def placement_in_polygon(size, contours, polygon_contour):
    # сортировка контуров в порядке убывания радиуса описанной окружности
    contours = sorted(contours, key=lambda x: cv.minEnclosingCircle(x)[1], reverse=True)
    # проверка на очевидную невозможность размещения
    polygon_radius = cv.minEnclosingCircle(polygon_contour)[1]
    polygon_area = cv.contourArea(polygon_contour)
    area_all_items = 0
    for contour in contours:
        item_area = cv.contourArea(contour)
        area_all_items += item_area
        if cv.minEnclosingCircle(contour)[1] > polygon_radius or item_area > polygon_area:
            return False
    if area_all_items > polygon_area:
        return False

    # размещение
    x, y, w, h = cv.boundingRect(polygon_contour)
    img_placement = np.zeros((size[0], size[1], 3), np.uint8)
    cv.fillPoly(img_placement, [polygon_contour], (255, 255, 255))
    step = int(min(w, h) / 10)
    step_angle = np.pi / 6
    for contour in contours:
        # перенос контура объекта к контуру многоугольника
        xc, yc, wc, hc = cv.boundingRect(contour)
        contour = move_contour(contour, -xc + x, -yc + y)
        answer, contour = try_place(x + int(wc / 2), y + int(hc / 2), x + w - int(wc / 2), y + h - int(hc / 2),
                                    img_placement, contour, step, step_angle)
        if answer == False:
            return False
        cv.fillPoly(img_placement, [contour], (255, 0, 0))
        plt.imshow(img_placement), plt.show()
    return True

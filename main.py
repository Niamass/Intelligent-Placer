import numpy as np
import cv2 as cv
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

'''
+ Нахождение многоугольника
+ Нахождение предметов
+ Идентификация предметов
- Размещение предметов в многоугольнике 
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
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # без этого drawContours не хочет работать
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
        x, y, width, height = cv.boundingRect(contours[i])
        img_obj = img[y - 17:y + height + 17, x - 17:x + width + 17]
        img_objects[i] = img_obj
        objects_coord[i] = (x, y, x + width, y + height)
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
    return img_with_contours, objects_coord, img_objects


# Поиск объектов на эталонном изображении (на фоне листа)
def get_reference_objects():
    objects_coord = {}
    img_objects = {}
    for i in range(0, 10):
        img = imread('Items/item_' + str(i) + '.jpg')
        min_color = [190, 190, 0]
        img_with_contour, object_contour = find_object_on_paper(img.copy(), min_color)
        obj, img_obj = find_coordinates_objects([object_contour], img)
        img_objects[i] = img_obj[0]
        objects_coord[i] = obj[0]
    return objects_coord, img_objects


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
    # лучшее совпадение - искомый объект
    correct_matches_dct = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    return list(correct_matches_dct.keys())[0]


# Поиск основных цветов на изображении с помощью кластеризации
def find_colors(img):
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans_clustering = KMeans(n_clusters=3, n_init=5, max_iter=100)
    kmeans_clustering.fit(img)
    centers = kmeans_clustering.cluster_centers_.astype("uint8").tolist()
    centers = sorted(centers, key=lambda x: x[0])
    return centers

# Поэлементное сравнение списков
def list_comp(a, b, c):
    for i in range(len(b)):
        if not (a[i] < b[i] < c[i]):
            return False
    return True

# Сравнение изображений найденных объектов с эталонными по преобладающим цветам для плохоопределяемых объектов
def search_matches_color(img_find_obj):
    find_obj_color = find_colors(img_find_obj)
    h, w = img_find_obj.shape[:2]
    for color in find_obj_color:
        if list_comp([70, 44, 23], color, [130, 80, 35]):
            return items_list[2]
        if list_comp([180, 45, 90], color, [187, 55, 100]):
            return items_list[1]
        if list_comp([150, 170, 40], color, [165, 185, 55]) and min(w, h) / max(w, h) <= 1.1:
            return items_list[0]
    return items_list[-1]


# Идентификация найденных объектов
def indentify_objects(img_find_objects, img_ref_objects):
    items = []
    for i in range(len(img_find_objects)):
        item = search_matches_color(img_find_objects[i])
        if item == 'not identified':
            item = search_matches_points(img_find_objects[i], img_ref_objects)
        items.append(item)
    return items


ref_objects_coord, img_ref_objects = get_reference_objects()
for i in range(14):
    print("----------------------------------------")
    print("test = ", i)
    img = imread('Tests/test_' + str(i) + '.jpg')
    img_with_contour, polygon_contour = find_polygon(img)
    img_with_contour, find_objects_coord, img_find_objects = find_objects(img_with_contour)
    plt.imshow(img_with_contour), plt.show()

    items = indentify_objects(img_find_objects, img_ref_objects)
    print(items)
import cv2
import os

# caminho para a imagem a ser recortada
img_path = 'Imagens\\European\\17.jpg'

#Carrega imagem e prepara para mostrar na tela
img_raw = cv2.imread(img_path)
h, w = img_raw.shape[:2]
h = int(h * 2.5)
w = int(w * 1)
img_raw = cv2.resize(img_raw, (h, w))

#função de bounding box (ROI) da bilbioteca OpenCV
rois = cv2.selectROIs('select', img_raw, False, False)

i = 0 #contador para nomear as imagens

#caminho onde as images recortadas serão salvas
path = 'Imagens\\Saudavel Crop'


for rect in rois:
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    img_crop = img_raw[y1:y1+y2, x1:x1+x2]
    #mostra as imagens recortadas na tela
    cv2.imshow("crop" + str(i), img_crop)
    #salva imagens recortadas com o nome "crop", segudi do valor do contador i, no formato jpg.
    cv2.imwrite(os.path.join(path, 'crop'+str(i)+'.jpg'),  img_crop)
    #incrementa o contador
    i += 1

cv2.waitKey(0)
#cv2.destroyAllWindows()
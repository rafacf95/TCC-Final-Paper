import cv2
import glob
import os

inputFolder = "americana" #caminho onde as imagens a serem reescaladas estão salvas
os.mkdir("americana\\novasImagens") #cria a pasta
outputFolder = "americana\\novasImagens" #caminho de destino para as novas imagens

folderLen = len(inputFolder)
i = 0
#percorre a pasta de origem
for img in glob.glob(inputFolder+"/*.jpg"):
    image = cv2.imread(img) #leva a imagem para a memoria
    imgResized = cv2.resize(image, (100, 100)) #remdimenciona a imagem
    cv2.imwrite(outputFolder+img[folderLen:], imgResized) #salva a imagem na pasta destino
    



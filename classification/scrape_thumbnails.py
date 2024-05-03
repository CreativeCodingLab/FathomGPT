# WEB Page Image Scraper
# https://github.com/agaraman0/Web_Page_Image_Scrapper/blob/master/web_page_scrapper.py

import requests
from bs4 import *
from PIL import Image
import matplotlib.pyplot as plt
import os

link = input("Input Your Link: ")
req = requests.get(link)
soup = BeautifulSoup(req.text,'lxml')
imgs=soup.find_all('img')
k = 1
for i in imgs:
    try:
        url =i['src']
        print('Image Link:',k)
        print(url)
        response = requests.get(url,stream=True)
        img = Image.open(response.raw)
        plt.imshow(img)
        plt.close()
        img.save('google_imgs/{}.jpg'.format(str(k)))
    except:
        KeyError
    k+=1
    
    if k > 10:
        break
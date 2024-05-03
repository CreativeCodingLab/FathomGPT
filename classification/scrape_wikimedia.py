# https://commons.wikimedia.org/wiki/Commons:Download_tools

import shutil
from lxml import etree
from lxml import html
import aiohttp
import asyncio
import aiofiles
import nest_asyncio
import os
import requests
import json

from fathomnet.api import boundingboxes

storeDirectory = 'wikimedia/'
checkForCategories = False
maxImgs = 100
    
tasks = []
categories = 0
categoryTasks = []
checkedCategories = []
completed = -1
totalImages = 0
completedImages = 0

def scrape(url):
    #fix for running in colab
    nest_asyncio.apply()

    #all categories will be compressed into storeDirectory + download.zip on completion

    #ONLY CHNAGE THIS
    #url = 'https://commons.wikimedia.org/wiki/Category:Neo-Baroque_interiors'
    #url = 'https://commons.wikimedia.org/wiki/Category:Acanthephyra_eximia'
    

    #DON'T CHANGE
    

    async def fetch_page(session, url, cat = ''):
      try:
        async with session.get(url) as resp:
          source = await resp.text()

          dom = html.fromstring(source)

          return [cat, dom]
      except asyncio.TimeoutError or aiohttp.ClientConnectorError:
        #print('Timeout')
        return False

    async def fetch_images(session, url):
      global totalImages

      dom = await fetch_page(session, url)

      #timeout error
      if dom == False:
        return

      images = dom[1].xpath('*//div[@class="thumb"]//a')
      images = images[:maxImgs]
      subcategories = dom[1].xpath('*//div[@class="CategoryTreeItem"]//a')

      if(len(subcategories) > 0 and checkForCategories):
        for category in subcategories:
          if(category not in checkedCategories):
            categoryTasks.append(asyncio.ensure_future(fetch_images(session, 'https://commons.wikimedia.org' + category.attrib['href'])))
            checkedCategories.append(category)
            print('Found category', category.attrib['href'])

      if (len(images) > 0):
        totalImages += len(images)
        print("Found", len(images), "images")
        #download images for each category
        for image in images:
          cat = url.split('Category:')[1]
          tasks.append(asyncio.ensure_future(fetch_page(session, 'https://commons.wikimedia.org' + image.attrib['href'], cat)))

      global completed
      completed += 1

    async def main(loop):
      global url
      global completedImages

      async with aiohttp.ClientSession(loop=loop) as session:
        


        await fetch_images(session, url)
        

        #fix to resolve finding all categories first
        while True:
          await asyncio.gather(*categoryTasks)

          #check if images have been found on all category pages
          if(completed == len(categoryTasks)):
            break

        pages = await asyncio.gather(*tasks)

        for page in pages:
          #timeout error
          if(page == False):
            continue

          cat = page[0]
          source = page[1]
           
          #print(cat, source.xpath('*//div[@class="fullImageLink"]//img')[0].attrib['src'])
          imgURL = source.xpath('*//div[@class="fullImageLink"]//img')[0].attrib['src']

          filename = imgURL.split('/')[-1]
          filename = str(completedImages)+'.jpg'
          #TODO: save images into category folders
          async with session.get(imgURL) as resp:
            if resp.status == 200:
                if(os.path.isdir(storeDirectory + cat + '/') == False):
                  os.mkdir(storeDirectory + cat + '/')

                f = await aiofiles.open(storeDirectory + cat + '/' + filename, mode='wb')
                await f.write(await resp.read())
                await f.close()
                completedImages += 1
                print(completedImages, '/', totalImages)

        completedImages = 0
        #tasks = []
            
            
      #create zip file to download
      #shutil.make_archive(storeDirectory + 'download', 'zip', storeDirectory)

    #main event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))



#categories = ['Acanthephyra eximia', 'Aurelia aurita']

categories = boundingboxes.find_concepts()[1:]
print(len(categories))
categories = [cat.replace(' ', '_') for cat in categories]

completedCat = []
with open('completed.json') as f:
    completedCat = json.load(f)
    
for category in categories:

    if category in completedCat:
        continue
    
    print(category)
    
    tasks = []
    categories = 0
    categoryTasks = []
    checkedCategories = []
    completed = -1
    totalImages = 0
    completedImages = 0

    url = 'https://commons.wikimedia.org/wiki/Category:'+category
    scrape(url)
    
    completedCat.append(category)
    
    with open("completed.json", "w") as outfile:
        json.dump(completedCat, outfile)
    
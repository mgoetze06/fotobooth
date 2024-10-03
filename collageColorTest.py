import tkinter as tk
from PIL import Image, ImageTk
import multiprocessing
import time
import os, random, shutil
from datetime import datetime
import cv2
from webserver.fotobooth_utils import writeImagecountToFile,writeCollageCountToFile,readRGBFromFile

def update_gallery(e): #collage process
    iteration = 0
    collagenumber = 0
    mode = 0
    while True:
        if e.is_set():
            e.clear()
            #print("try to create collage")
            directory = "/home/pi/programs/images/"
            folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
            
            files = folders = 0
            for _, dirnames, filenames in os.walk(folder):
              # ^ this idiom means "we won't be using this value"
                files += len(filenames)
                folders += len(dirnames)
            #print(files)
            if files > 3:
                imglist = []
                os.chdir(folder)
                imglist = os.listdir(os.getcwd())
                #print("now i create a collage")
                scr_w,scr_h = 1920,1080
                #print("full 2x2 collage without overlay")
                #overlay the png with stripes and logo on top of 2x2 collage
                cols = 2 #for full mode
                rows = 2
                new_img = Image.open("/home/pi/programs/countdown/overlay.jpg")
                new_img = new_img.resize((scr_w,scr_h),Image.ANTIALIAS)
                ims = []
                thumbnail_height = round(scr_h/rows)
                currentUsedPhotosInCollageList = []
                for i in range(0,4):
                    filename = folder + "/" + random.choice(imglist)
                    while (filename == folder + "/collages") or (filename in currentUsedPhotosInCollageList):
                        filename = folder + "/" + random.choice(imglist)

                    currentUsedPhotosInCollageList.append(filename)
                    image = Image.open(filename)

                    sizefactor = scr_h/image.height
                    thumbnail_width = round((image.width * sizefactor)/cols)
                    image = image.resize((thumbnail_width,thumbnail_height),Image.ANTIALIAS) #resize to new image size
                    ims.append(image)
            
                i = 0
                x = round((scr_w/cols)-thumbnail_width)
                y = round((scr_h/rows)-thumbnail_height)
                for col in range(cols):
                    for row in range(rows):
                        #print(i,x,y)
                        new_img.paste(ims[i], (x,y))
                        i += 1
                        y += thumbnail_height
                    x += thumbnail_width
                    y = 0
                image = Image.open("/home/pi/programs/countdown/stripes.png") #overlay the png with stripes and logo on top of 2x2 collage
                new_img.paste(image, (0,0), image) #second image is for alpha channel in foreground

                    
                if not os.path.exists(folder + "/collages/"):
                    os.makedirs(folder + "/collages/")
                    
                if collagenumber < 10:
                    name = folder + "/collages/collage-000" + str(collagenumber) + ".jpg"
                else:
                    if collagenumber < 100:
                        name = folder + "/collages/collage-00" + str(collagenumber) + ".jpg"
                    else:
                        if collagenumber < 1000:
                            name = folder + "/collages/collage-0" + str(collagenumber) + ".jpg"
                        else:                          
                            name = folder + "/collages/collage-" + str(collagenumber) + ".jpg"
                    
                new_img.save(name,'JPEG')
                collagenumber += 1
                try:
                    writeCollageCountToFile(collagenumber)
                except:
                    pass
                #mode += 1
                if mode > 2:
                    mode = 0
                time.sleep(60)
            
            
           #e.clear()

        else:
            time.sleep(0.2)
        

def createQuadraticCollage(size):
    #2x2
    scr_w,scr_h = 1920,1080
    cols = size #for full mode
    rows = size
    folder = "/home/pi/programs/images/folder1"
    files = ["C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG","C:\projects\\fotobooth\data\IMG_3944.JPG",
             "C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG",
             "C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG"]
    
    r,g,b = readRGBFromFile()
    new_img= Image.new(mode="RGBA", size=(scr_w,scr_h), color=(r,g,b,255))
    
    if len(files) < (size*size):
        return new_img


    #new_img = Image.open("C:\projects\\fotobooth\data\IMG_9012.JPG")
    new_img = new_img.resize((scr_w,scr_h))
    ims = []
    thumbnail_height = round(scr_h/rows)
    currentUsedPhotosInCollageList = []
    for i in range(0,4):
        filename = files[i]
        while (filename == folder + "/collages") or (filename in currentUsedPhotosInCollageList):
            filename = files[i]
        currentUsedPhotosInCollageList.append(filename)
        image = Image.open(filename)

        sizefactor = scr_h/image.height
        thumbnail_width = round((image.width * sizefactor)/cols)
        image = image.resize((thumbnail_width,thumbnail_height)) #resize to new image size
        ims.append(image)
        ims.append(image)
        ims.append(image)
    i = 0
    x = round((scr_w/cols)-thumbnail_width)
    y = round((scr_h/rows)-thumbnail_height)
    for col in range(cols):
        for row in range(rows):
            #print(i,x,y)
            new_img.paste(ims[i], (x,y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    return new_img


def createThreeStackedCollageWithBackground():
    scr_w,scr_h = 1920,1080
    cols = 2 #for full mode
    rows = 2
    folder = "/home/pi/programs/images/folder1"
    files = ["C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG","C:\projects\\fotobooth\data\IMG_3944.JPG"]
    
    r,g,b = readRGBFromFile()
    new_img= Image.new(mode="RGBA", size=(scr_w,scr_h), color=(r,g,b,255))

    ims = []
    stackedrows = 3
    thumbnail_height = round(scr_h/stackedrows)
    currentUsedPhotosInCollageList = []
    for i in range(0,stackedrows):
        filename = files[i]
        while (filename == folder + "/collages") or (filename in currentUsedPhotosInCollageList):
            filename = files[i]
        currentUsedPhotosInCollageList.append(filename)
        image = Image.open(filename)
        #print(str(image.width),str(image.height),str(image.height/image.width))
        sizefactor = scr_h/image.height
        thumbnail_width = round((image.width * sizefactor)/stackedrows)
        image = image.resize((thumbnail_width,thumbnail_height)) #resize to new image size
        ims.append(image)
    i = 0
    x = round(((scr_w*3)/4)-thumbnail_width/4)
    y = 0
    for i in range(0,stackedrows):
        new_img.paste(ims[i], (x,y))
        y += thumbnail_height

    return new_img

def createStackedCollagesOnBothSidesWithBackground():
    scr_w,scr_h = 1920,1080
    cols = 2 #for full mode
    rows = 2
    folder = "/home/pi/programs/images/folder1"
    files = ["C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG","C:\projects\\fotobooth\data\IMG_3944.JPG"]
    
    r,g,b = readRGBFromFile()
    new_img= Image.new(mode="RGBA", size=(scr_w,scr_h), color=(r,g,b,255))

    ims = []
    stackedrows = 3
    thumbnail_height = round(scr_h/stackedrows)
    currentUsedPhotosInCollageList = []
    for i in range(0,stackedrows):
        filename = files[i]
        while (filename == folder + "/collages") or (filename in currentUsedPhotosInCollageList):
            filename = files[i]
        currentUsedPhotosInCollageList.append(filename)
        image = Image.open(filename)
        #print(str(image.width),str(image.height),str(image.height/image.width))
        sizefactor = scr_h/image.height
        thumbnail_width = round((image.width * sizefactor)/stackedrows)
        image = image.resize((thumbnail_width,thumbnail_height)) #resize to new image size
        ims.append(image)
    i = 0
    x = round(scr_w-thumbnail_width)
    y = 0
    for i in range(0,stackedrows):
        new_img.paste(ims[i], (x,y))
        y += thumbnail_height
    
    i = 0
    x = 0
    y = 0
    for i in range(0,stackedrows):
        new_img.paste(ims[i], (x,y))
        y += thumbnail_height

    return new_img

if __name__ == '__main__':
    
    imglist = []
    show_last_two_photos = False
    lastfile = "asldfas"


    root = tk.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    #root.overrideredirect(1)
    root.geometry("%dx%d+0+0" % (w, h))
    root.overrideredirect(False)
   
    root.persistent_image = None
    root.attributes('-fullscreen',True)
    root.configure(background='black')
    
    #root.focus_set()    
    root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))
    canvas = tk.Canvas(root,width=w,height=h,highlightthickness=0)
   
    canvas.pack()
    canvas.configure(background='black')
    #root.overrideredirect(True)
    root.update()
    picwait_displayed = False
    pics_displayed = 0 #for collage display
    #imagepath =  "C:\projects\\fotobooth\data\IMG_3942.JPG"               
    #pilImage = Image.open(imagepath)
    pilImage = createQuadraticCollage(2)
    pilImage = createQuadraticCollage(3)

    pilImage = createThreeStackedCollageWithBackground()
    pilImage = createStackedCollagesOnBothSidesWithBackground()
    imgWidth, imgHeight = pilImage.size
    if imgWidth > w or imgHeight > h:
        ratio = min(w/imgWidth, h/imgHeight)
        imgWidth = int(imgWidth*ratio)
        imgHeight = int(imgHeight*ratio)
        pilImage = pilImage.resize((imgWidth,imgHeight))
    image = ImageTk.PhotoImage(pilImage)
    imagesprite = canvas.create_image(w/2,h/2,image=image)
    root.update()
    root.mainloop()
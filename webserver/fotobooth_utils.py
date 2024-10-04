import os
from glob import glob
import shutil
COLOR_FILE_NAME = "color.txt"
PHOTOS_FILE_NAME = "photos.txt"
COLLAGES_FILE_NAME = "collages.txt"
WEBSERVER_FOLDER = "/home/pi/programs/webserver"

def IsCustomCollageEnabled(folder):
    if os.path.isdir(os.path.join(folder,"customcollage")):
        return True
    else:
        return False

def enableCustomCollage(folder):
    os.mkdir(os.path.join(folder,"customcollage"))

def disableCustomCollage(folder):
    os.path.join(folder,"customcollage")
    print("trying to remove customcollage folder: ",folder)
    try:
        shutil.rmtree(folder)
    except:
        print("disableCustomCollage(): error removing customcollage folder")


def convertHexToTuple(hex):
     #    value is: #FF0000
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def convertTupleToHexString(rgb_tuple):
     return "#{:02x}{:02x}{:02x}".format(rgb_tuple[0],rgb_tuple[1],rgb_tuple[2])


def getFilenameWithRespectToWebserverDirectory(filename):
    if not os.path.isfile(filename):
        return os.path.join(WEBSERVER_FOLDER,filename)
    else:
        return filename

def clearOldCollages(collagesFolder):
    print("trying to remove collage folder: ",collagesFolder)
    if "collage" in collagesFolder:
        try:
            shutil.rmtree(collagesFolder)
        except:
            print("clearOldCollages(): error removing collages folder")

def writeRGBToFile(rgb_tuple):

    f = open(getFilenameWithRespectToWebserverDirectory(COLOR_FILE_NAME), "w") 
    rgb = "RGB"
    for i in range(3):
         f.write(rgb[i])
         f.write(str(rgb_tuple[i]))
         f.write('\n')
    f.close()


def readRGBFromFile():
    f = open(getFilenameWithRespectToWebserverDirectory(COLOR_FILE_NAME), "r") 
    rgb = "RGB"
    lines = f.readlines()
    rgb_tuple = []
    for i in range(3):
          rgb_tuple += [int(lines[i].lstrip(rgb[i]).replace("\n",""))]
    f.close()
    return rgb_tuple

def getImagecountFromFile():
    try:
        f = open(getFilenameWithRespectToWebserverDirectory(PHOTOS_FILE_NAME), "r") 
        lines = f.readlines()
        imagecount = lines[0].replace("\n","")
        f.close()
    except:
        imagecount = "Keine Datei gefunden."

    return imagecount

def countFilesInFolder(folder):
    try:
        imagecount = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
    except:
        imagecount = "Keine Dateien gefunden."

    return imagecount

def writeImagecountToFile(imagecount):
    try:
        f = open(getFilenameWithRespectToWebserverDirectory(PHOTOS_FILE_NAME), "w") 
        f.write(str(imagecount))
        f.close()
        return True
    except:
        return False
    


def getCollageCountFromFile():
    try:
        f = open(getFilenameWithRespectToWebserverDirectory(COLLAGES_FILE_NAME), "r") 
        lines = f.readlines()
        imagecount = lines[0].replace("\n","")
        f.close()
    except:
        imagecount = "Keine Datei gefunden."

    return imagecount

def writeCollageCountToFile(imagecount):
    try:
        f = open(getFilenameWithRespectToWebserverDirectory(COLLAGES_FILE_NAME), "w") 
        f.write(str(imagecount))
        f.close()
        return True
    except:
        return False
    

def getLatestFolder():
    directory = "/home/pi/programs/images/"
    print("searching directory: ", directory)
    try:
        folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
        print("found folder: ",folder)
    except:
        print("did not find ", directory)
        folder = 'C:\\projects\\fotobooth\\programs\\countdown'
        print("returning ", folder)
    return folder

def getLatestImage():
    folder = getLatestFolder()
    try:
        #os.chdir(folder)
        print("Searching latest image in imglist")
        #print(folder)
        imglist = glob(os.path.join(folder, '*'))
        #print(imglist)
        if len(imglist)>1:
            imglist = sorted(imglist, key=os.path.getmtime)
            #print(imglist)
        index = -1
        while(os.path.isdir(imglist[index])):
            index = index - 1
        return os.path.join(folder, imglist[index])
    except: 
        print("No image in imglist, returning default")
        if os.path.exists('/home/pi/programs/countdown'):
            return '/home/pi/programs/countdown/picwait.jpg'
        else:
            return 'C:\\projects\\fotobooth\\programs\\countdown\\example_collage.jpg'

    
#x = readRGBFromFile()
#print(x)
#newColor = convertTupleToHexString(x)
#print(newColor)
#print("imagecount:",getImagecountFromFile())
#print("imagecount Update erfolgreich:",writeImagecountToFile(27))
#print("imagecount:",getImagecountFromFile())
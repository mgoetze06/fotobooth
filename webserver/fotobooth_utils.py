
COLOR_FILE_NAME = "color.txt"
PHOTOS_FILE_NAME = "photos.txt"

def convertHexToTuple(hex):
     #    value is: #FF0000
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def convertTupleToHexString(rgb_tuple):
     return "#{:02x}{:02x}{:02x}".format(rgb_tuple[0],rgb_tuple[1],rgb_tuple[2])



def writeRGBToFile(rgb_tuple):
    f = open(COLOR_FILE_NAME, "w") 
    rgb = "RGB"
    for i in range(3):
         f.write(rgb[i])
         f.write(str(rgb_tuple[i]))
         f.write('\n')
    f.close()


def readRGBFromFile():
    f = open(COLOR_FILE_NAME, "r") 
    rgb = "RGB"
    lines = f.readlines()
    rgb_tuple = []
    for i in range(3):
          rgb_tuple += [int(lines[i].lstrip(rgb[i]).replace("\n",""))]
    f.close()
    return rgb_tuple

def getImagecountFromFile():
    try:
        f = open(PHOTOS_FILE_NAME, "r") 
        lines = f.readlines()
        imagecount = lines[0].replace("\n","")
        f.close()
    except:
        imagecount = "Keine Datei gefunden."

    return imagecount

def writeImagecountToFile(imagecount):
    try:
        f = open(PHOTOS_FILE_NAME, "w") 
        f.write(str(imagecount))
        f.close()
        return True
    except:
        return False
    
#x = readRGBFromFile()
#print(x)
#newColor = convertTupleToHexString(x)
#print(newColor)
#print("imagecount:",getImagecountFromFile())
#print("imagecount Update erfolgreich:",writeImagecountToFile(27))
#print("imagecount:",getImagecountFromFile())
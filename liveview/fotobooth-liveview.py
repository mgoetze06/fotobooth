#das problem an liveview ist der schwächere autofocus. Man kann die preview der Kamera zwar gut als vorschau verwenden
#z.b. um das positionieren vor dem auslösen vor der kamera zu ermöglichen
#bei umgeschaltenem Spiegel wird ein schwächerer autofocus verwendet, der bis zu 5s bfür die berechnung benötigt
#dadurch ist vor dem aufnehmen des bildes ein umschalten in den normalen dslr modus notwendig
#das umschalten des spiegels hört sich allerdings wie das aufnehmen eines bildes an, daher kommt es zur verwirrung


# --> liveview müsste also durch eine externe kamera realisiert werden




#import matplotlib.pyplot as plt
from PIL import Image,ImageTk
import io
import datetime
import gphoto2 as gp
import subprocess
import tkinter as tk
import os
import time
from PIL import ImageDraw,ImageFont
def subprocess_return(p,output,error):
    print(p.returncode)
    print(output)
    print(error)
    if p.returncode == 0:
        print("return code ok", output)
        #print('%r is found in %s: %r' % (pattern, filename, output))
    elif p.returncode == 1:
        print("return code 1", output)
        #print('%r is NOT found in %s: %r' % (pattern, filename, output))
    else:
        #assert p.returncode > 1
        print('error occurred: %r' % (error,))

def setCameraConfig(camera,configname,configvalue):
    #gp.check_result(gp.gp_camera_init(camera))
    # get configuration tree
    config = gp.check_result(gp.gp_camera_get_config(camera))
    # find the capture target config item
    capture_target = gp.check_result(
        gp.gp_widget_get_child_by_name(config, configname))

    # check value in range
    count = gp.check_result(gp.gp_widget_count_choices(capture_target))

    value = int(configvalue)

    if value < 0 or value >= count:
        print('Parameter out of range')
        return 1
    # set value
    value = gp.check_result(gp.gp_widget_get_choice(capture_target, value))
    gp.check_result(gp.gp_widget_set_value(capture_target, value))
    # set config
    gp.check_result(gp.gp_camera_set_config(camera, config))
    return 0


def cameraInit():
    subprocess.Popen(["pkill", "-f", "gphoto2"])
    camera = gp.check_result(gp.gp_camera_new())
    gp.check_result(gp.gp_camera_init(camera))
    # required configuration will depend on camera type!
    print('Checking camera config')
    # get configuration tree
    config = gp.check_result(gp.gp_camera_get_config(camera))
    # find the image format config item
    # camera dependent - 'imageformat' is 'imagequality' on some
    OK, image_format = gp.gp_widget_get_child_by_name(config, 'imageformat')
    if OK >= gp.GP_OK:
        # get current setting
        value = gp.check_result(gp.gp_widget_get_value(image_format))
        # make sure it's not raw
        if 'raw' in value.lower():
            print('Cannot preview raw images')
            exit
    # find the capture size class config item
    # need to set this on my Canon 350d to get preview to work at all
    OK, capture_size_class = gp.gp_widget_get_child_by_name(
        config, 'capturesizeclass')
    if OK >= gp.GP_OK:
        # set value
        print("setting capture size class")
        value = gp.check_result(gp.gp_widget_get_choice(capture_size_class, 4))
        gp.check_result(gp.gp_widget_set_value(capture_size_class, value))
        # set config
        gp.check_result(gp.gp_camera_set_config(camera, config))
    else:
        print("error setting capture size class")

    setCameraConfig(camera,'output',0)
    # capture preview image (not saved to camera memory card)
    print('Capturing preview image')
    #time.sleep(3)
    return camera

def convertCameraFileToPIL(camera_file):
    file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
    image_io = io.BytesIO(file_data)
    image = Image.open(image_io)
    return image

def getImageFromCamera(camera):
    camera_file = gp.check_result(gp.gp_camera_capture_preview(camera))
    image = convertCameraFileToPIL(camera_file)
    return image

def getNewImageName():
    directory = "/home/pi/programs/images/"
    folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
    now = datetime.datetime.now()
    newname = folder + "/IMG-" + now.strftime("%Y%m%d-%H%M%S") + ".jpg"
    print("this is newname: ")
    print(newname)
    return newname

def captureImage():
    print('Capturing image using pythongphoto')
    newname = getNewImageName()
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    #target = os.path.join('/tmp', file_path.name)
    #print('Copying image to', target)
    camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(newname)
    image = convertCameraFileToPIL(camera_file)
    return image

def captureImageSubprocess():
    print('Capturing image using subprocess and gphoto2')
    newname = getNewImageName()
    p2 = subprocess.Popen(["pkill", "-f", "gphoto2"])
    p2.wait()
    p1 = subprocess.Popen(["gphoto2", "--capture-image-and-download","--filename",newname,"--keep","--force-overwrite"])
    output, error = p1.communicate()
    subprocess_return(p1,output,error)
    p1.wait()

def updateCanvas(image,root,canvas):
    image = ImageTk.PhotoImage(image)
    imagesprite = canvas.create_image(w/2,h/2,image=image)
    root.update()

def resizeImageToCanvas(pilImage,w,h):
    imgWidth, imgHeight = pilImage.size
    if imgWidth > w or imgHeight > h:
        ratio = min(w/imgWidth, h/imgHeight)
        imgWidth = int(imgWidth*ratio)
        imgHeight = int(imgHeight*ratio)
        pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
    
    return pilImage

def addTextToImage(image,i,imax):
    text = str(i)+" / " +str(imax)
    #font = ImageFont.truetype("sans-serif.ttf", 32)
    ImageDraw.Draw(
        image  # Image
    ).text(
        (10, 10),  # Coordinates
        text,  # Text
        (0, 0, 0)#,font=font  # Color
    )

    return image



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
camera = cameraInit()

imax = 15
for i in range(imax):
    a = datetime.datetime.now()
    image = getImageFromCamera(camera)
    print(image.size)
    image = addTextToImage(image,i,imax)
    image = resizeImageToCanvas(image,w,h)
    updateCanvas(image,root,canvas)
    b = datetime.datetime.now()
    print("processed in %s ms"%((b-a).total_seconds()*1000))

#camera.exit()
#captureImageSubprocess()

image = captureImage()
image = resizeImageToCanvas(image,w,h)
updateCanvas(image,root,canvas)
time.sleep(10)
camera.exit()
import logging
try:
    import RPi.GPIO as GPIO
except Exception as ex:
    GPIO = None
    logging.warning("RPi.GPIO unavailable: %s", ex)

import tkinter as tk
from PIL import Image, ImageTk
import multiprocessing
import time
import os, random, shutil
import subprocess
from subprocess import check_output
import psutil
try:
    from gpiozero import CPUTemperature
except Exception as ex:
    CPUTemperature = None
    print("gpiozero unavailable: {}".format(ex))
from datetime import datetime
import cv2
import io
try:
    import gphoto2 as gp
except Exception as ex:
    gp = None
    logging.warning("gphoto2 library unavailable: %s", ex)
from webserver.fotobooth_utils import writeImagecountToFile,writeCollageCountToFile,readRGBFromFile,IsCustomCollageEnabled,getCountdownFromFile,getSleepTimeSecondsFromFile,getShowSingleImageAlwaysWithOverlay

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ENABLE_OLED = True
ENABLE_WLED = True
OLED_DEVICE = None
ws = None
leds = None
channel = None

def try_init_oled():
    global OLED_DEVICE
    try:
        from luma.core.interface.serial import i2c
        from luma.core.render import canvas
        from luma.oled.device import sh1106
        serial = i2c(port=1, address=0x3C)
        OLED_DEVICE = sh1106(serial)
        return True
    except Exception as ex:
        logger.warning("OLED disabled: %s", ex)
        OLED_DEVICE = None
        return False


def try_init_wled():
    global ws, leds, channel
    try:
        import _rpi_ws281x as _ws
        ws = _ws
        leds = ws.new_ws2811_t()
        return True
    except Exception as ex:
        logger.warning("WLED disabled: %s", ex)
        ws = None
        leds = None
        return False


LED_CHANNEL    = 0
LED_COUNT      = 24         # How many LEDs to light.
LED_FREQ_HZ    = 800000     # Frequency of the LED signal.  Should be 800khz or 400khz.
LED_DMA_NUM    = 5          # DMA channel to use, can be 0-14.
LED_GPIO       = 12         # GPIO connected to the LED signal line.  Must support PWM!
LED_BRIGHTNESS = 150        # Set to 0 for darkest and 255 for brightest
LED_INVERT     = 0          # Set to 1 to invert the LED signal, good if using NPN
                            # transistor as a 3.3V->5V level converter.  Keep at 0
                            # for a normal/non-inverted signal
DOT_COLORS = [0x922b21,
              0xb03a2e,
              0x6c3483,
              0x2874a6,
              0x148f77,
              0xd4ac0d,
              0xd35400,
              0x581845,
              0x900c3f,
              0xc70039,
              0xff5733,
              0x509916,
              0x00b2f0,
              0xc06c84,
              0xf8b195]




        
def update_oled(e):
    if not ENABLE_OLED or OLED_DEVICE is None:
        logger.info("OLED disabled, OLED process will sleep.")
        while True:
            time.sleep(1)

    iteration = 0
    photo_count = 0
    updated = False
    logger.info("updating oled process started. Waiting for update events")
    try:
        with open('/home/pi/programs/log_backup.txt', 'r') as f:
            lastbackup = f.read()
    except Exception:
        lastbackup = ""

    from luma.core.render import canvas

    while True:
        if animation_finished.is_set() and updated == False:
            photo_count += 1
            updated = True
            with canvas(OLED_DEVICE) as draw:
                draw.rectangle(OLED_DEVICE.bounding_box, outline="white", fill="black")
                draw.text((10, 4), "heute aufgenommen: ", fill=1)
                draw.text((50, 30), str(photo_count), fill=1)
            try:
                writeImagecountToFile(photo_count)
            except Exception as ex:
                logger.warning("update_oled(): writeImagecountToFile failed: %s", ex)
                
        if not animation_finished.is_set():
            updated = False
            
        if e.is_set():
            print("update oled")
            with canvas(OLED_DEVICE) as draw:
                draw.rectangle(OLED_DEVICE.bounding_box, outline="white", fill="black")
                if iteration == 5:
                    draw.text((10, 4), "heute aufgenommen: ", fill=1)
                    draw.text((50, 26), str(photo_count), fill=1)

                if iteration == 1:
                    draw.text((3, 26), "party-fotobox@web.de", fill=1)

                if iteration == 2:
                    if CPUTemperature is not None:
                        cpu = CPUTemperature()
                        load = round((cpu.temperature/85)*100,2)
                        draw.text((6, 4), "CPU Temperatur: ", fill=1)
                        draw.text((50, 16), str(round(cpu.temperature,2)) +" °C", fill=1)
                        draw.text((6, 28), "CPU Load Percentage: ", fill=1)
                        draw.text((50, 40), str(load) +" %", fill=1)
                    else:
                        draw.text((6, 4), "CPU Temperatur: n/a", fill=1)

                if iteration == 3:
                    disk = psutil.disk_usage('/')
                    disk_free = round((disk.free /2**30),2)
                    disk_total = round((disk.total /2**30),2)
                    disk_percentage = round((disk_free /disk_total)*100)
                    draw.text((6, 4), "Freier Speicher: ", fill=1)
                    draw.text((20, 16), str(disk_free) +" GB ("+str(disk_percentage)+"%)", fill=1)
                    draw.text((6, 28), "Gesamtspeicher: ", fill=1)
                    draw.text((20, 40), str(disk_total) +" GB", fill=1)

                if iteration == 4:
                    draw.text((10, 4), "Last Backup: ", fill=1)
                    draw.text((10, 26), lastbackup, fill=1)

                if iteration == 0:
                    try:
                        ssid = subprocess.check_output(['iwgetid']).decode().strip()
                        ip = check_output(['hostname', '-I']).decode().strip()
                    except Exception as ex:
                        logger.warning("update_oled(): network info unavailable: %s", ex)
                        ssid = "FOTOBOX"
                        ip = ""
                    draw.text((10, 4), ssid, fill=1)
                    draw.text((10, 26), ip, fill=1)
            e.clear()
            iteration += 1
        else:
            time.sleep(0.2)
        if iteration == 6:
            iteration = 0

        
def detectFaces():
    print("detecting faces")
    

def createQuadraticCollage(size,imagepaths,folder):
    #2x2
    scr_w,scr_h = 1920,1080
    cols = size #for full mode
    rows = size
    #folder = "/home/pi/programs/images/folder1"
    #files = ["C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG","C:\projects\\fotobooth\data\IMG_3944.JPG",
    #         "C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG",
    #         "C:\projects\\fotobooth\data\IMG_9012.JPG","C:\projects\\fotobooth\data\IMG_9019.JPG","C:\projects\\fotobooth\data\IMG_3942.JPG"]
    
    files = imagepaths
    try:
        r,g,b = readRGBFromFile()
        new_img= Image.new(mode="RGB", size=(scr_w,scr_h), color=(r,g,b))
    except Exception as ex:
        logger.warning("createQuadraticCollage(): failed to read RGB, using black background: %s", ex)
        new_img= Image.new(mode="RGB", size=(scr_w,scr_h), color=(0,0,0))

    if len(files) < (size*size):
        return new_img
    
    try:
        new_img = new_img.resize((scr_w,scr_h))
    except Exception as ex:
        logger.warning("createQuadraticCollage(): resize fallback used: %s", ex)
        new_img = new_img.resize((scr_w,scr_h),Image.ANTIALIAS)
    ims = []
    thumbnail_height = round(scr_h/rows)
    currentUsedPhotosInCollageList = []
    for i in range(0,4):
        try:
            randchoice = random.choice(imagepaths)
        except Exception as ex:
            logger.warning("createQuadraticCollage(): no random image choice available: %s", ex)
            return None
        filename = folder + "/" + randchoice
        while (filename == folder + "/collages") or (filename == folder + "/customcollage") or (filename in currentUsedPhotosInCollageList):
            try:
                randchoice = random.choice(imagepaths)
            except Exception as ex:
                logger.warning("createQuadraticCollage(): failed to choose next image: %s", ex)
                return None
            filename = folder + "/" + randchoice
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

    overlay = readOverlay("overlayQuadratic.png")
    new_img = overlayImage(new_img,overlay)
    return new_img

def createCustomCollageWithThreeImagesOnRightSide(imagepaths,folder):
    scr_w,scr_h = 1920,1080

    files = imagepaths
    try:
        filename = os.path.join(folder,"customcollage")
        filename = os.path.join(filename,"custom.jpg")
        new_img= Image.open(filename)
    except Exception as ex:
        logger.warning("createCustomCollageWithThreeImagesOnRightSide(): custom collage file unavailable: %s", ex)
        new_img= Image.new(mode="RGB", size=(scr_w,scr_h), color=(0,0,0))
    ims = []
    stackedrows = 3
    if len(files) < (stackedrows):
        return new_img
    thumbnail_height = round(scr_h/stackedrows)
    currentUsedPhotosInCollageList = []
    for i in range(0,stackedrows):
        try:
            randchoice = random.choice(imagepaths)
        except Exception as ex:
            logger.warning("createCustomCollageWithThreeImagesOnRightSide(): no random image choice available: %s", ex)
            return None
        filename = folder + "/" + randchoice
        while (filename == folder + "/collages") or (filename == folder + "/customcollage") or (filename in currentUsedPhotosInCollageList):
            try:
                randchoice = random.choice(imagepaths)
            except Exception as ex:
                logger.warning("createCustomCollageWithThreeImagesOnRightSide(): failed to choose next image: %s", ex)
                return None
            filename = folder + "/" + randchoice
        currentUsedPhotosInCollageList.append(filename)
        image = Image.open(filename)
        sizefactor = scr_h/image.height
        thumbnail_width = round((image.width * sizefactor)/stackedrows)
        try:
            image = image.resize((thumbnail_width,thumbnail_height)) 
        except Exception as ex:
            logger.warning("createCustomCollageWithThreeImagesOnRightSide(): resize fallback used: %s", ex)
            image = image.resize((thumbnail_width,thumbnail_height),Image.ANTIALIAS)
        ims.append(image)
    i = 0
    x = round(((scr_w*3)/4)-thumbnail_width/4)
    y = 0
    for i in range(0,stackedrows):
        new_img.paste(ims[i], (x,y))
        y += thumbnail_height

    return new_img


def update_gallery(e): #collage process
    iteration = 0
    collagenumber = 0
    mode = 0
    while True:
        if e.is_set():
            e.clear()
            logger.info("try to create collage")
            directory = "/home/pi/programs/images/"
            try:
                subfolders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
                folder = max([os.path.join(directory,d) for d in subfolders], key=os.path.getmtime)
            except Exception as ex:
                logger.warning("update_gallery(): cannot find latest image folder: %s", ex)
                time.sleep(1)
                continue
            files = folders = 0
            for _, dirnames, filenames in os.walk(folder):
              # ^ this idiom means "we won't be using this value"
                files += len(filenames)
                folders += len(dirnames)
            if files > 3:
                imglist = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
                try:
                    if mode == 0:
                        new_img = createQuadraticCollage(2,imglist,folder)
                    elif mode == 1 and IsCustomCollageEnabled(folder):
                        new_img = createCustomCollageWithThreeImagesOnRightSide(imglist,folder)
                    else:
                        new_img = createQuadraticCollage(2,imglist,folder)
                except Exception as ex:
                    logger.warning("update_gallery(): collage creation failed: %s", ex)
                    new_img = None
                if new_img:
                    collages_folder = os.path.join(folder, "collages")
                    os.makedirs(collages_folder, exist_ok=True)
                    name = os.path.join(collages_folder, "collage-{0:04d}.jpg".format(collagenumber))
                    try:
                        new_img.save(name, 'JPEG')
                        collagenumber += 1
                    except Exception as ex:
                        logger.warning("update_gallery(): failed saving collage: %s", ex)
                    try:
                        writeCollageCountToFile(collagenumber)
                    except Exception as ex:
                        logger.warning("update_gallery(): writeCollageCountToFile failed: %s", ex)
                    mode = (mode + 1) % 2
                time.sleep(60)
            else:
                logger.info("not enough images for collage in folder: %s", folder)
        else:
            time.sleep(0.2)
        

def timerfunc(e):      #timer for updating oled display and gallery on main display
    while True:#e is animation finished
        gallerytime = 7     #time between new photos are shown on main display
        multiplikator = 2   #gallerytime * multiplikator = time elapsed before oled display gets updated
        start_fresh = False
        start = time.time()
        end = time.time()
        
        for i in range(multiplikator - 1):  #run the timer 3 times to 5s before oled gets updated
            while(end - start)<(gallerytime *(i+1)):
                time.sleep(0.01)
                if first_button_pushed.is_set() or animation_finished.is_set():
                        print("timerfunc is going to sleep")
                        time.sleep(gallerytime*2)
                        print("timerfunc woke up")
                        start_fresh = True
                        break
                end = time.time()
            if not e.is_set() and not first_button_pushed.is_set():
                gallery_update_event.set()     #set here and clear event in gallery process
        if not start_fresh:
            oled_update_event.set()            #set here and clear event in oled process

def resetGphoto2():
    subprocess.Popen(["pkill", "-f", "gphoto2"])
    time.sleep(1)

def take_photo(e):
    while True:
        if e.is_set():
            print('take photo...')
            # subprocess.Popen(["pkill", "-f", "gphoto2"])
            # p2 = subprocess.Popen(["gphoto2","--set-config","capturetarget=1"])
            # output, error = p2.communicate()
            # tries = 1
            # while not error == None and tries < 5:
            #     p2 = subprocess.Popen(["gphoto2","--set-config","capturetarget=1"])
            #     output, error = p2.communicate()
            #     tries += 1
            # subprocess_return(p2,output,error)
            # p2.wait()
            # #time.sleep(0.3)
            # directory = "/home/pi/programs/images/"
            # folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
            # now = datetime.now()
            # newname = folder + "/IMG-" + now.strftime("%Y%m%d-%H%M%S") + ".jpg"
            # print("this is newname: ")
            # print(newname)
            # p1 = subprocess.Popen(["gphoto2", "--capture-image-and-download","--filename",newname,"--keep","--force-overwrite"])
            # #p1 = subprocess.Popen(["gphoto2", "--capture-image-and-download","--filename","/home/pi/programs/images/new.jpg","--force-overwrite"])
            # #time.sleep(1)
            # output, error = p1.communicate()
            # subprocess_return(p1,output,error)
            # tries = 1
            # while not error == None and tries < 5:
            #     p1 = subprocess.Popen(["gphoto2","--set-config","capturetarget=1"])
            #     output, error = p2.communicate()
            #     tries += 1
            # p1.wait()
            # time.sleep(0.1)
            # first_button_pushed.clear()
            # e.clear()
            # photo_taken_event.set()
            #oled_update_event.set()            #set here and clear event in oled process

            #files = folders = 0
            #for _, dirnames, filenames in os.walk("/home/pi/programs/images/"):
              # ^ this idiom means "we won't be using this value"
                #files += len(filenames)
                #folders += len(dirnames)
            #directory = "/home/pi/programs/images/"
            #folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
            #if nr < 10:
            #    newname = folder + "/IMG-000" + str(nr) + ".jpg"
            #else:
            #    if nr < 100:
            #        newname = folder + "/IMG-00" + str(nr) + ".jpg"
            #    else:
            #        if nr < 1000:
            #            newname = folder + "/IMG-0" + str(nr) + ".jpg"
            #        else:
            #            newname = folder + "/IMG-" + str(nr) + ".jpg"
            #now = datetime.now()
            #newname = folder + "/IMG-" + now.strftime("%Y%m%d-%H%M%S") + ".jpg"
            #print("this is newname bevore copying: ")
            #print(newname)
            #shutil.copy("/home/pi/programs/newimage/new.jpg", newname)
            #nr += 1
def startWebserver():
    try:
        subprocess.Popen(["python","./webserver/fotobooth_webserver.py"],cwd="/home/pi/programs")
    except Exception as ex:
        logger.warning("startWebserver(): failed to start webserver: %s", ex)

def readSleepTimeSecondsFromFile():
    try:
        return getSleepTimeSecondsFromFile()
    except Exception as ex:
        logger.warning("readSleepTimeSecondsFromFile(): %s", ex)
        return 0.1

def led_countdown(e): #e is first button pushed
    iteration = 0
    offset = 0 #offset for countdown animation
    offset_idle = 0 #offset for idle animation
    old_color = 0x0
    new_color = 0x1
    old_ring = True
    def wheel(pos):
        if pos < 0 or pos > 255:
            r = g = b = 0
        elif pos < 85:
            r = int(pos * 3)
            g = int(255 - pos * 3)
            b = 0
        elif pos < 170:
            pos -= 85
            r = int(255 - pos * 3)
            g = 0
            b = int(pos * 3)
        else:
            pos -= 170
            r = 0
            g = int(pos * 3)
            b = int(255 - pos * 3)
            
        r = "{:02x}".format(r) 
        g = "{:02x}".format(g) 
        b = "{:02x}".format(b)
        combined = "0x"+r+g+b
        hex_int = int(combined, 16)
        #print(hex_int)
        return hex_int

    def rainbow_cycle(wait):
        for j in range(255):
            for i in range(LED_COUNT):
                pixel_index = (i * 256 // LED_COUNT) + j
                #pixels[i] = wheel(pixel_index & 255)
                color = wheel(pixel_index & 255)
                ws.ws2811_led_set(channel, i, color)
                resp = ws.ws2811_render(leds)
                
            #pixels.write()
            time.sleep(wait)
    sleepTimeRefresher = 0
    sleepTimeSeconds = readSleepTimeSecondsFromFile()

    if not ENABLE_WLED or ws is None or channel is None:
        print("WLED disabled, led_countdown will simulate button flow.")
        while True:
            sleepTimeRefresher += 1
            if sleepTimeRefresher == 500:
                sleepTimeRefresher = 0
                sleepTimeSeconds = readSleepTimeSecondsFromFile()
                print("Updating sleeptime to: ",sleepTimeSeconds)

            if e.is_set() and not animation_finished.is_set():
                print('simulating countdown without WLED...')
                if offset == LED_COUNT//3 or offset == 0 or offset == 2*LED_COUNT//3:
                    print("setting animation breakpoint")
                    animation_breakpoint.set()
                    time.sleep(sleepTimeSeconds)
                offset += 1
                print("offset: ",offset)
                time.sleep(sleepTimeSeconds)
                if offset == LED_COUNT + 1:
                    offset = 0
                    animation_breakpoint.set()
                    animation_finished.set()
                    animation_breakpoint.clear()
                    offset_idle = round(LED_COUNT/4)
                    first_button_pushed.clear()
                    time.sleep(4)
            else:
                time.sleep(0.1)
        # end simulated WLED loop

    while True:
        sleepTimeRefresher += 1
        if sleepTimeRefresher == 500:
            sleepTimeRefresher = 0
            sleepTimeSeconds = readSleepTimeSecondsFromFile()
            print("Updating sleeptime to: ",sleepTimeSeconds)
        while e.is_set() and not animation_finished.is_set():
            print('animating countdown leds ...')
            if offset == LED_COUNT//3 or offset == 0 or offset == 2*LED_COUNT//3:
                print("setting animation breakpoint")
                animation_breakpoint.set()
                time.sleep(sleepTimeSeconds)

            for i in range(LED_COUNT):
                if i < offset:
                    color = 0xffffff #white
                else:
                    color = 0x000000 #black

                # Set the LED color buffer value.
                ws.ws2811_led_set(channel, i, color)
                # Send the LED color data to the hardware.
                resp = ws.ws2811_render(leds)
                # Increase offset to animate colors moving.  
            offset += 1
            print("offset: ",offset)

            time.sleep(sleepTimeSeconds)
            if offset == LED_COUNT + 1:
                offset = 0
                animation_breakpoint.set()
                #time.sleep(0.3)
                animation_finished.set()
                animation_breakpoint.clear()
                offset_idle = round(LED_COUNT/4)
                first_button_pushed.clear()
                time.sleep(4)
#               
        #end of while
                
        if not e.is_set() and animation_finished.is_set(): #first button not pushed
            print("ledprocess: first button pushed; animation finished")
            for i in range(LED_COUNT):
                color = 0xffffff #black
                ws.ws2811_led_set(channel, i, color)
            resp = ws.ws2811_render(leds)
            
        if not e.is_set() and not animation_finished.is_set(): #led animation for idle
            #print("ledprocess: led idle, should animate")
            if old_ring:
                new_color = DOT_COLORS[iteration]
                for i in range(LED_COUNT):
                    if i < offset_idle:
                        color = new_color
                    else:
                        color = old_color
                    ws.ws2811_led_set(channel, i, color)
                    resp = ws.ws2811_render(leds)
                offset_idle += 1
                time.sleep(0.001)
                if offset_idle == LED_COUNT + 1:
                    iteration += 1
                    offset_idle = 0
                    #offsetidle = random.randint(0,23)
                    old_color = new_color
                if iteration == len(DOT_COLORS):
                    iteration = 0
            else:
                rainbow_cycle(0.000000001)

def convertCameraFileToPIL(camera_file):
    if gp is None:
        raise RuntimeError('libgphoto2 Python bindings are unavailable')
    file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
    image_io = io.BytesIO(file_data)
    image = Image.open(image_io)
    return image

def getNewImageName():
    directory = "/home/pi/programs/images/"
    folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
    now = datetime.now()
    newname = folder + "/IMG-" + now.strftime("%Y%m%d-%H%M%S") + ".jpg"
    print("this is newname: ")
    print(newname)
    return newname

def captureImage(camera):
    if not camera:
        camera = cameraInit()

    image = None
    try:
        print('Capturing image using pythongphoto')
        newname = getNewImageName()
        file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
        print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
        #target = os.path.join('/tmp', file_path.name)
        #print('Copying image to', target)
        camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
        camera_file.save(newname)
        image = convertCameraFileToPIL(camera_file)
    except:
        print("captureImage(): error capturing photo")
    return image

def readOverlay(filename_without_path):
    try:
        directory = "/home/pi/programs/images/"
        subfolders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if not subfolders:
            raise FileNotFoundError('No image folders found')
        folder = max([os.path.join(directory, d) for d in subfolders], key=os.path.getmtime)
        filename = os.path.join(folder, "customcollage", filename_without_path)
        return Image.open(filename)
    except Exception as ex:
        logger.warning("readOverlay(): overlay file not available: %s", ex)
        return None


def resizeImageToCanvasWithOverlay(pilImage,w,h,overlayFilename):
    try:

        overlayImage = readOverlay(overlayFilename)
        if overlayImage is None:
            return pilImage
        imgWidth, imgHeight = pilImage.size
        image = pilImage
        if imgWidth > w:
            ratio = min(w/imgWidth, 1)
            imgWidth = int(imgWidth*ratio)
            imgHeight = int(imgHeight*ratio)
            image = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
        new_img = Image.new(mode="RGB", size=(scr_w,scr_h), color=(0,0,0))
        new_img.paste(image, (round(scr_w/2-imgWidth/2), scr_h-imgHeight))
        if overlayImage is not None:
            new_img.paste(overlayImage,(0,0),overlayImage)
        return new_img
    except Exception as ex:
        logger.warning("error setting overlay image: %s", ex)
        return pilImage


def overlayImage(sourceImage,overlay):
    try:
        sourceImage.paste(overlay,(0,0),overlay)
    except:
        print("error setting overlay")
    return sourceImage

def resizeImageToCanvas(pilImage,w,h):
    imgWidth, imgHeight = pilImage.size
    if imgWidth > w or imgHeight > h:
        ratio = min(w/imgWidth, h/imgHeight)
        imgWidth = int(imgWidth*ratio)
        imgHeight = int(imgHeight*ratio)
        pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)

        try:
            r,g,b = readRGBFromFile()
            new_img = Image.new(mode="RGB", size=(scr_w,scr_h), color=(r,g,b))
            new_img.paste(pilImage, (round(scr_w/2-imgWidth/2),0))
            pilImage = new_img
        except Exception as ex:
            logger.warning("error setting background: %s", ex)
    return pilImage

def cameraInit():
    subprocess.Popen(["pkill", "-f", "gphoto2"])
    if gp is None:
        logger.warning('cameraInit(): libgphoto2 Python bindings unavailable, skipping camera init')
        return None
    try:
        camera = gp.check_result(gp.gp_camera_new())
        gp.check_result(gp.gp_camera_init(camera))
    except Exception as ex:
        logger.warning("cameraInit(): error initializing camera: %s", ex)
        return None
    logger.info('Checking camera config')
    try:
        config = gp.check_result(gp.gp_camera_get_config(camera))
    except Exception as ex:
        logger.warning("cameraInit(): error getting camera config: %s", ex)
        return None
    OK, image_format = gp.gp_widget_get_child_by_name(config, 'imageformat')
    if OK >= gp.GP_OK:
        value = gp.check_result(gp.gp_widget_get_value(image_format))
        if 'raw' in value.lower():
            logger.warning('Cannot preview raw images')
            return None
    # find the capture size class config item
    # need to set this on my Canon 350d to get preview to work at all
    OK, capture_size_class = gp.gp_widget_get_child_by_name(
        config, 'capturesizeclass')
    if OK >= gp.GP_OK:
        logger.info("setting capture size class")
        value = gp.check_result(gp.gp_widget_get_choice(capture_size_class, 4))
        gp.check_result(gp.gp_widget_set_value(capture_size_class, value))
        gp.check_result(gp.gp_camera_set_config(camera, config))
    else:
        logger.warning("error setting capture size class")

    #setCameraConfig(camera,'output',0)
    # capture preview image (not saved to camera memory card)
    #print('Capturing preview image')
    time.sleep(1)
    return camera

def listImages():
    global imglist
    global folder
    try:
        imglist = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if len(imglist) > 1:
            imglist = sorted(imglist, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    except Exception as ex:
        logger.warning("listImages(): failed to list images in %s: %s", folder, ex)
        imglist = []
    #print("listimages: ",imglist)
    #oldest = files[0]
    #newest = files[-1]


def randImg(pics_displayed,show_last_two_photos_local,lastfile):
    global imglist
    global show_last_two_photos
    global folder
    #global lastfile
    listImages()
    collagelist = []
    myimage = "/home/pi/programs/countdown/picwait.jpg"
    if imglist == []:
        print("no file found")
        myimage = "/home/pi/programs/countdown/picwait.jpg"
    else:
        if pics_displayed == 4 and os.path.exists(os.path.join(folder, "collages")):
            logger.info("displaying collage")
            collages_folder = os.path.join(folder, "collages")
            try:
                collagelist = [f for f in os.listdir(collages_folder) if os.path.isfile(os.path.join(collages_folder, f))]
                collagelist = sorted(collagelist, key=lambda f: os.path.getmtime(os.path.join(collages_folder, f)))
                myimage = random.choice(collagelist)
                myimage = os.path.join(collages_folder, myimage)
                logger.info("Selected collage: %s", myimage)
            except Exception as ex:
                logger.warning("randImg(): error accessing existing collages: %s", ex)
                myimage = "/home/pi/programs/countdown/picwait.jpg"
            
        else:
            if pics_displayed < 3 and len(imglist) > 2 and show_last_two_photos:
                logger.info("displaying gallery after new photo")
                index = (-1 * (pics_displayed + 1)) - 1
                if index > -4:
                    myimage = imglist[index]
                    while myimage == "collages":
                        index -= 1
                        if index < -len(imglist):
                            break
                        myimage = imglist[index]
                else:
                    myimage = imglist[-1]
                    show_last_two_photos = False
                myimage = os.path.join(folder, myimage)
                logger.info("Selected recent image: %s", myimage)
            else:
                try:
                    myimage = random.choice(imglist)
                except Exception:
                    myimage = "/home/pi/programs/countdown/picwait.jpg"
                while (myimage == lastfile) or (myimage == "collages"):
                    try:
                        myimage = random.choice(imglist)
                    except Exception:
                        myimage = "/home/pi/programs/countdown/picwait.jpg"
                lastfile = myimage
                if "picwait" not in myimage:
                    myimage = os.path.join(folder, myimage)
                logger.info("Selected random image: %s", myimage)
                show_last_two_photos = False
    return myimage


def newImg():
    global imglist
    try:
        debug = True
        listImages()
        if not imglist:
            raise FileNotFoundError('No images available')
        myimage = imglist[-1]
        now = datetime.now()
        if debug:
            logger.info("now: %s", now)
            logger.info("latest image: %s", myimage)
        image_date = myimage.split("IMG-")[1].split(".jpg")[0]
        image_date_astime = datetime.strptime(image_date, "%Y%m%d-%H%M%S")
        if debug:
            logger.info("image datetime: %s", image_date_astime)
        failedCounter = 0
        while (now - image_date_astime).total_seconds() > 15:
            if debug:
                logger.info("image too old: %s", (now - image_date_astime).total_seconds())
            failedCounter += 1
            if failedCounter > 10:
                raise FileNotFoundError
            time.sleep(0.5)
            listImages()
            if not imglist:
                raise FileNotFoundError('No images available')
            myimage = imglist[-1]
            now = datetime.now()
            image_date = myimage.split("IMG-")[1].split(".jpg")[0]
            image_date_astime = datetime.strptime(image_date, "%Y%m%d-%H%M%S")
        logger.info("Returning new image: %s", myimage)
        return myimage
    except Exception as ex:
        logger.warning("newImg(): %s", ex)
        return "/home/pi/programs/countdown/picwait.jpg"

def getCountdownImageFromCounter(counter):
    path = "/home/pi/programs/countdown/picwait.jpg"
    if counter < 0 or counter > 3:
        return path

    if counter == 0:
        path = "/home/pi/programs/countdown/pic1.jpg"
    if counter == 1:
        path = "/home/pi/programs/countdown/pic2.jpg"
    if counter == 2:
        path = "/home/pi/programs/countdown/pic3.jpg"
    if counter == 3:
        path = "/home/pi/programs/countdown/picwait.jpg"

    return path

def readRuediger():
    path = "/home/pi/programs/countdown/ruediger.jpg"
    return path

def reactToRuedigerDisplayed():
    global camera
    resetGphoto2()
    camera = cameraInit()
    numberOfAttempts = 0
    while camera == None and numberOfAttempts < 10:
        time.sleep(5)
        resetGphoto2()
        camera = cameraInit()
        numberOfAttempts += 1

def creation_date(path_to_file):
#"""
#Try to get the date that a file was created, falling back to when it was
#last modified if that isn't possible.
#See http://stackoverflow.com/a/39501288/1709587 for explanation.
#"""
    stat = os.stat(path_to_file)
    try:
        return stat.st_birthtime
    except AttributeError:
        # We're probably on Linux. No easy way to get creation dates here,
        # so we'll settle for when its content was last modified.
        #return time.ctime(stat.st_mtime)
        return datetime.fromtimestamp(stat.st_mtime)

def checkAndCreateFolder(parent_path,new_folder):
    if not parent_path.endswith("/"):
        parent_path = parent_path + "/"
    folder_to_check = parent_path + new_folder
    if os.path.exists(folder_to_check):
        print(parent_path + " contains " + new_folder +" already.")
    else:
        print(parent_path + " does not contain " + new_folder)
        os.makedirs(folder_to_check)
        print("directory " + new_folder + " created.")
    print(os.listdir(folder_to_check))
    folders = [name for name in os.listdir(folder_to_check) if os.path.isdir(os.path.join(folder_to_check, name))]


    folders = len(folders)

    print("amount of folders")
    print(folders)
    folder = "/home/pi/programs/images/folder" + str(folders)
    print("last folder: ", folder)
    folderdate = creation_date(folder)
    print("folder creation time: ", folderdate)
    currenttime = datetime.fromtimestamp(time.time())
    print("current time: ", currenttime)
    timediff_minutes = abs((folderdate - currenttime).total_seconds()/60) #timediff in minutes
    print(timediff_minutes)

    #foldertime check is working as intended, but raspberry is not having correct time due to lack of rtc module
    #therefore always create new folder
    # if(timediff_minutes > 60*12):
    #     folder = "/home/pi/programs/images/folder" + str(folders + 1)
    #     print("old folder. need to create new one: ", folder)
    #     while os.path.exists(folder):
    #         folders += 1
    #         folder = "/home/pi/programs/images/folder" + str(folders)
    #     os.makedirs(folder)
    # else:
    #     print("folder is not old enough. reuse folder: ", folder)
    createNewFolder = False
    if createNewFolder:
        folder = "/home/pi/programs/images/folder" + str(folders + 1)
        print("old folder. need to create new one: ", folder)
        while os.path.exists(folder):
            folders += 1
            folder = "/home/pi/programs/images/folder" + str(folders)
        os.makedirs(folder)


    return folder

def readCountdownFromFile():
    try:
        return getCountdownFromFile()
    except Exception as ex:
        logger.warning("readCountdownFromFile(): %s", ex)
        return False
    
def readShowSingleImageAlwaysWithOverlay():
    return False
    # try:
    #     return getShowSingleImageAlwaysWithOverlay()
    # except:
    #     return False

if __name__ == '__main__':
    oled_available = try_init_oled()
    wled_available = try_init_wled()

    if wled_available:
        leds = ws.new_ws2811_t()

        # Initialize all channels to off
        for channum in range(2):
            channel = ws.ws2811_channel_get(leds, channum)
            ws.ws2811_channel_t_count_set(channel, 0)
            ws.ws2811_channel_t_gpionum_set(channel, 0)
            ws.ws2811_channel_t_invert_set(channel, 0)
            ws.ws2811_channel_t_brightness_set(channel, 0)

        channel = ws.ws2811_channel_get(leds, LED_CHANNEL)
        ws.ws2811_channel_t_count_set(channel, LED_COUNT)
        ws.ws2811_channel_t_gpionum_set(channel, LED_GPIO)
        ws.ws2811_channel_t_invert_set(channel, LED_INVERT)
        ws.ws2811_channel_t_brightness_set(channel, LED_BRIGHTNESS)

        ws.ws2811_t_freq_set(leds, LED_FREQ_HZ)
        ws.ws2811_t_dmanum_set(leds, LED_DMA_NUM)

        resp = ws.ws2811_init(leds)
        if resp != 0:
            print('ws2811_init failed with code {0}'.format(resp))
            wled_available = False
            leds = None
            channel = None
    else:
        print('Skipping WLED initialization because hardware is unavailable.')

    if GPIO is not None:
        GPIO.setmode(GPIO.BCM)
        button1_pin = 23
        GPIO.setup(button1_pin, GPIO.IN)
        button2_pin = 24
        GPIO.setup(button2_pin, GPIO.IN)
        led_pin = 25
        GPIO.setup(led_pin, GPIO.OUT)
        backup_vcc = 8
        GPIO.setup(backup_vcc, GPIO.OUT)
        GPIO.output(backup_vcc, 1)
    else:
        print('Skipping GPIO setup because RPi.GPIO is unavailable.')

    try:
        with open('/home/pi/programs/log_backup.txt', 'r') as f:
            lastbackup = f.read()
    except Exception as ex:
        logger.warning("no backup file found: %s", ex)

    if not oled_available:
        logger.info('Skipping OLED initialization because hardware is unavailable.')

    first_button_pushed = multiprocessing.Event()
    animation_finished = multiprocessing.Event()
    gallery_update_event = multiprocessing.Event()
    oled_update_event = multiprocessing.Event()
    photo_taken_event = multiprocessing.Event()
    animation_breakpoint = multiprocessing.Event()

    def but1_callback(channel):
        print('first button pushed')
        first_button_pushed.set()

    def but2_callback(channel):
        print('backup button pushed')
        print('test: backing up now')

    if GPIO is not None:
        GPIO.add_event_detect(button1_pin, GPIO.RISING, callback=but1_callback, bouncetime=300)
        GPIO.add_event_detect(button2_pin, GPIO.FALLING, callback=but2_callback, bouncetime=300)
    else:
        print('GPIO event detection disabled because RPi.GPIO is unavailable.')

    process_timer = multiprocessing.Process(name='timer_process', target=timerfunc, args=(animation_finished,))
    process_gallery = multiprocessing.Process(name='gallery_process', target=update_gallery, args=(gallery_update_event,))
    process_led_count = multiprocessing.Process(name='first_process', target=led_countdown, args=(first_button_pushed,))

    process_gallery.daemon = True
    process_gallery.start()

    if oled_available:
        process_oled = multiprocessing.Process(name='oled_process', target=update_oled, args=(oled_update_event,))
        process_oled.daemon = True
        process_oled.start()
    else:
        print('OLED process not started.')

    process_timer.daemon = True
    process_timer.start()
    process_led_count.daemon = True
    process_led_count.start()


    #directory = "/home/pi/programs/images/"
    #folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
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
    imagesprite = canvas.create_image(w/2, h/2, image=None)
    root.image = None
    #root.overrideredirect(True)
    root.update()
    
    startWebserver()
    scr_w = 1920
    scr_h = 1080
    #create new folder 
    folder = checkAndCreateFolder("/home/pi/programs","images")
    print(folder)
    #shutil.copy("/home/pi/programs/images/fendt.jpeg", folder+"/fendt.jpeg")
    #imagepath = "/home/pi/programs/images/test.jpg"
    picwait_displayed = False
    ignoreOtherEvents = False
    newimage = False
    pics_displayed = 0 #for collage display
    animation_breakpoint_counter = 0 
    camera = cameraInit()
    overlayImage = None
    ruedigerDisplayed = False

    def display_image(pilImage, use_overlay):
        try:
            if use_overlay:
                pilImage = resizeImageToCanvasWithOverlay(pilImage, w, h, "overlay.png")
            else:
                pilImage = resizeImageToCanvas(pilImage, w, h)
        except Exception as ex:
            print("display_image(): resize failed: {}".format(ex))
        image = ImageTk.PhotoImage(pilImage)
        canvas.itemconfig(imagesprite, image=image)
        root.image = image
        root.update()

    showCountdown = readCountdownFromFile()
    showSingleImageAlwaysWithOverlay = readShowSingleImageAlwaysWithOverlay()
    showCountdownRefresher = 0
    while True:
        imagechanged = False
        ignoreOverlay = True

        if showCountdownRefresher >= 1000:
            showCountdownRefresher = 0
            showCountdown = readCountdownFromFile()
            showSingleImageAlwaysWithOverlay = readShowSingleImageAlwaysWithOverlay()

        showCountdownRefresher += 1

        if animation_finished.is_set():
            animation_breakpoint.clear()
            animation_finished.clear()
            first_button_pushed.clear()
            animation_breakpoint_counter = 0
            numberOfCaptureTries = 0
            pilImage = captureImage(camera)
            while pilImage is None and numberOfCaptureTries < 3:
                pilImage = captureImage(camera)
                numberOfCaptureTries += 1
                time.sleep(0.5)

            if pilImage is not None:
                imagechanged = True
                gallery_update_event.clear()
                picwait_displayed = False
                pics_displayed = 0
                show_last_two_photos = True
                newimage = True
            else:
                print("BITTE NICHT SO NAH RAN RÜDIGER")
                imagepath = readRuediger()
                imagechanged = True
                ruedigerDisplayed = True

        elif first_button_pushed.is_set() and not picwait_displayed and not showCountdown:
            imagepath = getCountdownImageFromCounter(-1)
            imagechanged = True
            show_last_two_photos = True
            picwait_displayed = True
            print("picwait")

        elif animation_breakpoint.is_set() and showCountdown:
            imagepath = getCountdownImageFromCounter(animation_breakpoint_counter)
            imagechanged = True
            show_last_two_photos = True
            picwait_displayed = True
            animation_breakpoint.clear()
            animation_breakpoint_counter += 1
            print(imagepath)

        elif gallery_update_event.is_set():
            print("updating gallery")
            ignoreOverlay = False
            imagepath = randImg(pics_displayed, show_last_two_photos, lastfile)
            lastfile = imagepath
            imagechanged = True
            pics_displayed += 1
            if pics_displayed == 5:
                ignoreOverlay = True
                pics_displayed = 0

        if imagechanged:
            if not newimage:
                try:
                    pilImage = Image.open(imagepath)
                except Exception as ex:
                    print("main loop: failed to open image '{}': {}".format(imagepath, ex))
                    imagepath = randImg(1, False)
                    pilImage = Image.open(imagepath)

            display_image(pilImage, showSingleImageAlwaysWithOverlay and not ignoreOverlay)
            gallery_update_event.clear()
            if ruedigerDisplayed:
                reactToRuedigerDisplayed()
                ruedigerDisplayed = False
            newimage = False

        else:
            time.sleep(0.05)
    

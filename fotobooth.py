import RPi.GPIO as GPIO
import tkinter as tk
from PIL import Image, ImageTk
import multiprocessing
import time
import os, random, shutil
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1106
import _rpi_ws281x as ws
import subprocess
from subprocess import check_output
import psutil
from gpiozero import CPUTemperature
from datetime import datetime
import cv2
from webserver.fotobooth_utils import writeImagecountToFile,writeCollageCountToFile,readRGBFromFile,IsCustomCollageEnabled


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
    iteration = 0
    photo_count = 0
    updated = False
    print("updating oled process started. Waiting for update events")

    with open('/home/pi/programs/log_backup.txt', 'r') as f:
        lastbackup = f.read()
        f.close()

    while True:
        if animation_finished.is_set() and updated == False:
            photo_count += 1
            updated = True
            with canvas(device) as draw:
                draw.rectangle(device.bounding_box, outline="white", fill="black")
                draw.text((10, 4), "heute aufgenommen: ", fill=1)
                draw.text((50, 30), str(photo_count), fill=1) #(horizontal von links, vertikal von oben)
            try:
                writeImagecountToFile(photo_count)
            except:
                pass
                
        if not animation_finished.is_set():
            updated = False
            
        if e.is_set():
            print("update oled")
            with canvas(device) as draw:
                    draw.rectangle(device.bounding_box, outline="white", fill="black")
                    if iteration == 5:
                        
                        draw.text((10, 4), "heute aufgenommen: ", fill=1)
                        draw.text((50, 26), str(photo_count), fill=1) #(horizontal von links, vertikal von oben)


                    
                    if iteration == 1:
                        draw.text((3, 26), "party-fotobox@web.de", fill=1)
                        
                    if iteration == 2:
                        cpu = CPUTemperature()
                        load = round((cpu.temperature/85)*100,2)
                        draw.text((6, 4), "CPU Temperatur: ", fill=1)
                        draw.text((50, 16), str(round(cpu.temperature,2)) +" °C", fill=1)
                        draw.text((6, 28), "CPU Load Percentage: ", fill=1)
                        draw.text((50, 40), str(load) +" %", fill=1) 
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
                            ssid = subprocess.check_output(['iwgetid']).decode()
                            ip = check_output(['hostname', '-I'])
                        except:
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
    except:
        new_img= Image.new(mode="RGB", size=(scr_w,scr_h), color=(0,0,0))

    if len(files) < (size*size):
        return new_img
    
    try:
        new_img = new_img.resize((scr_w,scr_h))
    except:
        new_img = new_img.resize((scr_w,scr_h),Image.ANTIALIAS)
    ims = []
    thumbnail_height = round(scr_h/rows)
    currentUsedPhotosInCollageList = []
    for i in range(0,4):
        try:
            randchoice = random.choice(imagepaths)
        except:
            return None
        filename = folder + "/" + randchoice
        while (filename == folder + "/collages") or (filename == folder + "/customcollage") or (filename in currentUsedPhotosInCollageList):
            try:
                randchoice = random.choice(imagepaths)
            except:
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

    return new_img

def createCustomCollageWithThreeImagesOnRightSide(imagepaths,folder):
    scr_w,scr_h = 1920,1080

    files = imagepaths
    try:
        filename = os.path.join(folder,"customcollage")
        filename = os.path.join(filename,"custom.jpg")
        new_img= Image.open(filename)
    except:
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
        except:
            return None
        filename = folder + "/" + randchoice
        while (filename == folder + "/collages") or (filename == folder + "/customcollage") or (filename in currentUsedPhotosInCollageList):
            try:
                randchoice = random.choice(imagepaths)
            except:
                return None
            filename = folder + "/" + randchoice
        currentUsedPhotosInCollageList.append(filename)
        image = Image.open(filename)
        sizefactor = scr_h/image.height
        thumbnail_width = round((image.width * sizefactor)/stackedrows)
        try:
            image = image.resize((thumbnail_width,thumbnail_height)) 
        except:
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
            print("try to create collage")
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
                #try:
                if mode == 0:
                    new_img = createQuadraticCollage(2,imglist,folder)
                else:
                    if mode == 1:
                        if IsCustomCollageEnabled(folder):
                            new_img = createCustomCollageWithThreeImagesOnRightSide(imglist,folder)
                        else:
                            new_img = createQuadraticCollage(2,imglist,folder)
                if new_img:
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
                    #except:
                    #    print("update_gallery(): error creating collage")

                    try:
                        writeCollageCountToFile(collagenumber)
                    except:
                        pass
                    mode += 1
                    if mode > 1:
                        mode = 0
                time.sleep(60)
            print("not enough images for collage in folder:",folder)
            
           #e.clear()

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
                time.sleep(0.1)
                if photo_taken_event.is_set() or first_button_pushed.is_set() or animation_finished.is_set():
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



def take_photo(e):
    #nr = 1
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
    while True:
        if e.is_set():
            print('take photo...')
            subprocess.Popen(["pkill", "-f", "gphoto2"])
            p2 = subprocess.Popen(["gphoto2","--set-config","capturetarget=1"])
            output, error = p2.communicate()
            tries = 1
            while not error == None and tries < 5:
                p2 = subprocess.Popen(["gphoto2","--set-config","capturetarget=1"])
                output, error = p2.communicate()
                tries += 1
            subprocess_return(p2,output,error)
            p2.wait()
            #time.sleep(0.3)
            directory = "/home/pi/programs/images/"
            folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
            now = datetime.now()
            newname = folder + "/IMG-" + now.strftime("%Y%m%d-%H%M%S") + ".jpg"
            print("this is newname: ")
            print(newname)
            p1 = subprocess.Popen(["gphoto2", "--capture-image-and-download","--filename",newname,"--keep","--force-overwrite"])
            #p1 = subprocess.Popen(["gphoto2", "--capture-image-and-download","--filename","/home/pi/programs/images/new.jpg","--force-overwrite"])
            #time.sleep(1)
            output, error = p1.communicate()
            subprocess_return(p1,output,error)
            tries = 1
            while not error == None and tries < 5:
                p1 = subprocess.Popen(["gphoto2","--set-config","capturetarget=1"])
                output, error = p2.communicate()
                tries += 1
            p1.wait()
            first_button_pushed.clear()
            e.clear()
            photo_taken_event.set()
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
        else:
            time.sleep(0.2)
def startWebserver():
    try:
        subprocess.Popen(["python","./webserver/fotobooth_webserver.py"],cwd="/home/pi/programs")

    except:
        pass

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
    
    while True:
        while e.is_set() and not animation_finished.is_set():
            print('animating countdown leds ...')
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
            if offset == LED_COUNT + 1:
                offset = 0
                animation_finished.set()
                offset_idle = round(LED_COUNT/4)
#                 
        #end of while
                
        if not e.is_set() and photo_taken_event.is_set(): #first button not pushed
            for i in range(LED_COUNT):
                color = 0xffffff #black
                ws.ws2811_led_set(channel, i, color)
            resp = ws.ws2811_render(leds)
            
        if not photo_taken_event.is_set() and not e.is_set() and not animation_finished.is_set(): #led animation for idle
            #print("led idle, should animate")
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
                time.sleep(0.01)
                if offset_idle == LED_COUNT + 1:
                    iteration += 1
                    offset_idle = 0
                    #offsetidle = random.randint(0,23)
                    old_color = new_color
                if iteration == len(DOT_COLORS):
                    iteration = 0
            else:
                rainbow_cycle(0.0000001)
            



if __name__ == '__main__':
    
    
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

    # Initialize library with LED configuration.
    resp = ws.ws2811_init(leds)
    if resp != 0:
        raise RuntimeError('ws2811_init failed with code {0}'.format(resp))

    # initialize GPIO buttons
    #capture button
    GPIO.setmode(GPIO.BCM)
    button1_pin = 23
    GPIO.setup(button1_pin, GPIO.IN)
    
    #Button Backup
    button2_pin = 24
    GPIO.setup(button2_pin, GPIO.IN)
    #LED Backup
    led_pin = 25
    GPIO.setup(led_pin, GPIO.OUT)
    #Backup VCC
    backup_vcc = 8
    GPIO.setup(backup_vcc, GPIO.OUT)
    GPIO.output(backup_vcc,1)
    
    
    
    with open('/home/pi/programs/log_backup.txt', 'r') as f:
        lastbackup = f.read()
        f.close()

    #waiting for i2c service to start
    time.sleep(15)
    serial = i2c(port=1, address=0x3C)
    # substitute ssd1331(...) or sh1106(...) below if using that device
    device = sh1106(serial)
    #with canvas(device) as draw:
    #        draw.rectangle(device.bounding_box, outline="white", fill="black")
    #        draw.text((4, 4), "Setup; Last Backup:", fill=1)
    #        draw.text((4,30), lastbackup, fill=1)

    first_button_pushed = multiprocessing.Event()
    animation_finished = multiprocessing.Event()
    gallery_update_event = multiprocessing.Event()    
    oled_update_event = multiprocessing.Event()
    photo_taken_event = multiprocessing.Event()

    # GPIO callbacks
    def but1_callback(channel):
        print('first button pushed')
        #if first_button_pushed.is_set():
            #first_button_pushed.clear()
        #else:
        first_button_pushed.set()
    # GPIO callbacks
    def but2_callback(channel):
        
        ##just for testin
        
        print('backup button pushed')
        #GPIO.output(led_pin,1)
        #ledcounter = 0
        #for ledcounter in range (10):
        #    GPIO.output(led_pin,1)
        #    time.sleep(0.25)
        #    GPIO.output(led_pin,0)
        #    time.sleep(0.25)
        print('test: backing up now')
        #subprocess.call("/home/pi/programs/usbbackup/backup.sh")    
        #start bash script to backup/copy data here
        
        #start bash
    

    # GPIO callbacks hooks
    GPIO.add_event_detect(button1_pin, GPIO.RISING, callback=but1_callback, bouncetime=300)
    # GPIO callbacks hooks
    GPIO.add_event_detect(button2_pin, GPIO.FALLING, callback=but2_callback, bouncetime=300)


    # a process used to run the "long_processing" function in background
    # the first_button_pushed event is passed along
    process_led_count = multiprocessing.Process(name='first_process', target=led_countdown, args=(first_button_pushed,)) #led countdown
    process_camera = multiprocessing.Process(name='camera_process', target=take_photo, args=(animation_finished,))
    process_timer = multiprocessing.Process(name='timer_process', target=timerfunc, args=(animation_finished,))
    process_gallery = multiprocessing.Process(name='gallery_process', target=update_gallery, args=(gallery_update_event,))
    process_oled = multiprocessing.Process(name='oled_process', target=update_oled, args=(oled_update_event,))

    
    process_gallery.daemon = True
    process_gallery.start()
    
    process_oled.daemon = True
    process_oled.start()
    
    process_timer.daemon = True
    process_timer.start()
    
    process_camera.daemon = True
    process_camera.start()
    
    process_led_count.daemon = True
    process_led_count.start()


    #directory = "/home/pi/programs/images/"
    #folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
    imglist = []
    show_last_two_photos = False
    lastfile = "asldfas"
    def listImages():
        global imglist
        global folder
        os.chdir(folder)
        #imglist = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
        imglist = [f for f in os.listdir(os.getcwd()) if os.path.isfile(os.path.join(folder, f))]
        if len(imglist)>1:
            imglist = sorted(imglist, key=os.path.getmtime)
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
        if imglist == []:
            print("no file found")
            myimage = "/home/pi/programs/countdown/picwait.jpg"
        else:
            if pics_displayed == 4 and os.path.exists(folder + "/collages/"):
                #display Collage
                print("displaying collage")
                os.chdir(folder + "/collages/")
                collagelist = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
                try:
                    myimage = random.choice(collagelist)
                    print(myimage)
                except:
                    print("randImg(): error accessing existing collages")
                
            else:
                if pics_displayed < 3  and len(imglist) > 2 and show_last_two_photos == True:
                    #display last two images
                    print("displaying gallery after new foto")
                    #if len(imglist) > 2:
                        
                    index = (-1 * (pics_displayed + 1)) - 1
                    print(index)
                    if index > -4:
                        myimage = imglist[index]
                        while(myimage == "collages"):
                            #myimage = random.choice(os.listdir(self.imagepath))
                            index -= 1
                            myimage = imglist[index]
                            print(myimage)
                    else:
                        myimage = imglist[-1]
                        show_last_two_photos = False
                    myimage = folder + "/" + myimage
                    print(myimage)
                    #else:
                        
                else:
                    #random image
                    try:
                        myimage = random.choice(imglist)
                    except:
                        myimage = "/home/pi/programs/countdown/picwait.jpg"

                    
                    while (myimage == lastfile) or (myimage == "collages"):
                        #myimage = random.choice(os.listdir(self.imagepath))
                        try:
                            myimage = random.choice(imglist)
                        except:
                            myimage = "/home/pi/programs/countdown/picwait.jpg"

                    lastfile = myimage
                    if "picwait" not in myimage:
                        myimage = folder + "/" + myimage
                    print("random image")
                    print(myimage)
                    print(show_last_two_photos)
                    show_last_two_photos = False
        return myimage
    

    def newImg():
        global imglist
        try:
            debug = True
            listImages()
            myimage = imglist[-1]
            
            now = datetime.now()
            if debug:
                print("now: ",now)
                print(myimage)
            image_date = myimage.split("IMG-")[1].split(".jpg")[0]
            image_date_astime = datetime.strptime(image_date,"%Y%m%d-%H%M%S")
            if debug:
                print("image: ",image_date_astime)
            failedCounter = 0
            while (now - image_date_astime).total_seconds() > 15:
                if debug:
                    print((now - image_date_astime).total_seconds())
                    print("image too old")
                    failedCounter += 1
                    if failedCounter > 10:
                        raise FileNotFoundError
                time.sleep(0.5)
                listImages()
                myimage = imglist[-1]
                
                now = datetime.now()
                if debug:
                    print("now: ",now)
                image_date = myimage.split("IMG-")[1].split(".jpg")[0]
                image_date_astime = datetime.strptime(image_date,"%Y%m%d-%H%M%S")
                if debug:
                    print("image: ",image_date_astime)

            
            #newname = folder + "/IMG-" + now.strftime("%Y%m%d-%H%M%S") + ".jpg"
            
            #myimage = "/home/pi/programs/newimage/new.jpg"
            print(myimage)
            return myimage
        except:
            print("Fetching new Image failed.")
            print("Returning Wait-Image.")
            return "/home/pi/programs/countdown/picwait.jpg"
        
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
    
    startWebserver()
    scr_w = 1920
    scr_h = 1080
    #create new folder 
    folder = checkAndCreateFolder("/home/pi/programs","images")
    print(folder)
    #shutil.copy("/home/pi/programs/images/fendt.jpeg", folder+"/fendt.jpeg")
    #imagepath = "/home/pi/programs/images/test.jpg"
    picwait_displayed = False
    pics_displayed = 0 #for collage display
    while True:
        if photo_taken_event.is_set():
            imagepath = newImg()
            print("found new photo")
            imagechanged = True
            gallery_update_event.clear()
            picwait_displayed = False
            pics_displayed = 0
            show_last_two_photos = True #flag to show last two photos
        else:
            if first_button_pushed.is_set() and not picwait_displayed == True:
                imagepath = "/home/pi/programs/countdown/picwait.jpg"
                imagechanged = True
                show_last_two_photos = True #flag to show last two photos
                picwait_displayed = True
                print("picwait")
            else:
                if gallery_update_event.is_set():
                    print("updating gallery")
                    imagepath = randImg(pics_displayed,show_last_two_photos,lastfile)
                    lastfile = imagepath
                    imagechanged = True
                    pics_displayed += 1
                    if pics_displayed == 5:
                        pics_displayed = 0

                else:
                    imagechanged = False
        if imagechanged == True:
            try:
                pilImage = Image.open(imagepath)
            except:
                imagepath = randImg(1,False)
                pilImage = Image.open(imagepath)
            imgWidth, imgHeight = pilImage.size
            if imgWidth > w or imgHeight > h:
                ratio = min(w/imgWidth, h/imgHeight)
                imgWidth = int(imgWidth*ratio)
                imgHeight = int(imgHeight*ratio)
                pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)

                try:
                    r,g,b = readRGBFromFile()
                    new_img= Image.new(mode="RGB", size=(scr_w,scr_h), color=(r,g,b))
                    new_img.paste(pilImage, (round(scr_w/2-imgWidth/2),0))
                    pilImage = new_img
                except:
                    print("error setting background")
            image = ImageTk.PhotoImage(pilImage)
            imagesprite = canvas.create_image(w/2,h/2,image=image)
            root.update()
            gallery_update_event.clear()
            photo_taken_event.clear()
    

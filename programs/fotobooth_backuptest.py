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
import psutil
from gpiozero import CPUTemperature


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




def backup_usb():
    #os.system("/usr/bin/python3 /home/pi/programs/usbbackup/backup_usb.py")
    os.system("sudo /home/pi/programs/backup.sh")
    #subprocess.Popen(["/usr/bin/python3","/home/pi/programs/usbbackup/backup_usb.py"])



        
def update_oled(e):
    iteration = 0
    photo_count = 0
    updated = False
    while True:
        if animation_finished.is_set() and updated == False:
            photo_count += 1
            updated = True
            with canvas(device) as draw:
                draw.rectangle(device.bounding_box, outline="white", fill="black")
                draw.text((10, 4), "heute aufgenommen: ", fill=1)
                draw.text((50, 30), str(photo_count), fill=1) #(horizontal von links, vertikal von oben)
                
        if not animation_finished.is_set():
            updated = False
            
        if e.is_set():
            print("update oled")
            with canvas(device) as draw:
                    draw.rectangle(device.bounding_box, outline="white", fill="black")
                    if iteration == 0:
                        
                        draw.text((10, 4), "heute aufgenommen: ", fill=1)
                        draw.text((50, 26), str(photo_count), fill=1) #(horizontal von links, vertikal von oben)
                    
                    if iteration == 1:
                        draw.text((3, 26), "party-fotobox@web.de", fill=1)
                        
                    if iteration == 2:
                        cpu = CPUTemperature()
                        draw.text((10, 4), "CPU Temperatur: ", fill=1)
                        draw.text((50, 26), str(cpu.temperature) +" °C", fill=1) 
                    if iteration == 3:
                        disk = psutil.disk_usage('/')
                        disk_free = round((disk.free /2**30),2)
                        draw.text((10, 4), "Freier Speicher: ", fill=1)
                        draw.text((50, 26), str(disk_free) +" GB", fill=1) 
                
            e.clear()
            iteration += 1
        if iteration == 4:
            iteration = 0
        
    
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
                if mode == 0:
                    #print("full 2x2 collage without overlay")
                    #overlay the png with stripes and logo on top of 2x2 collage
                    cols = 2 #for full mode
                    rows = 2
                    new_img = Image.open("/home/pi/programs/countdown/overlay.jpg")
                    new_img = new_img.resize((scr_w,scr_h),Image.ANTIALIAS)
                    ims = []
                    thumbnail_height = round(scr_h/rows)
                    for i in range(0,4):
                        filename = folder + "/" + random.choice(imglist)
                        while filename == folder + "/collages":
                            filename = folder + "/" + random.choice(imglist)
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
                if mode == 1: #3 images on the right, left is place for logo or individual photo 
                    new_img = Image.open("/home/pi/programs/countdown/overlay.jpg")
                    new_img = new_img.resize((scr_w,scr_h),Image.ANTIALIAS)
                    ims = []
                    stackedrows = 3
                    thumbnail_height = round(scr_h/stackedrows)
                    for i in range(0,stackedrows):
                        filename = folder + "/" + random.choice(imglist)
                        while filename == folder + "/collages":
                            filename = folder + "/" + random.choice(imglist)
                        image = Image.open(filename)
                        #print(str(image.width),str(image.height),str(image.height/image.width))
                        sizefactor = scr_h/image.height
                        thumbnail_width = round((image.width * sizefactor)/stackedrows)
                        image = image.resize((thumbnail_width,thumbnail_height),Image.ANTIALIAS) #resize to new image size
                        ims.append(image)
                    i = 0
                    x = round(((scr_w*2)/3)-thumbnail_width/4)
                    y = 0
                    for i in range(0,stackedrows):
                        new_img.paste(ims[i], (x,y))
                        y += thumbnail_height
                    
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
                mode += 1
                if mode > 1:
                    mode = 0
                time.sleep(120)
            
            
           #e.clear()
        

def timerfunc(e):      #timer for updating oled display and gallery on main display
    while True:#e is animation finished
        gallerytime = 4     #time between new photos are shown on main display
        multiplikator = 3   #gallerytime * multiplikator = time elapsed before oled display gets updated
        
        start = time.time()
        end = time.time()
        
        for i in range(multiplikator - 1):  #run the timer 3 times to 5s before oled gets updated
            while(end - start)<(gallerytime *(i+1)):
                time.sleep(0.5)
                if photo_taken_event.is_set() or first_button_pushed.is_set() or animation_finished.is_set():
                        time.sleep(15)
                end = time.time()
            if not e.is_set() and not first_button_pushed.is_set():
                gallery_update_event.set()     #set here and clear event in gallery process
        oled_update_event.set()            #set here and clear event in oled process



def take_photo(e):
    nr = 1
    while True:
        if e.is_set():
            print('take photo...')
            subprocess.Popen(["pkill", "-f", "gphoto2"])
            subprocess.Popen(["gphoto2","--set-config","capturetarget=1"])
            time.sleep(0.2)
            p1 = subprocess.Popen(["gphoto2", "--capture-image-and-download","--filename","/home/pi/programs/newimage/new.jpg","--keep","--force-overwrite"])
            #p1 = subprocess.Popen(["gphoto2", "--capture-image-and-download","--filename","/home/pi/programs/images/new.jpg","--force-overwrite"])
            #time.sleep(1)
            p1.wait()
            first_button_pushed.clear()
            e.clear()
            photo_taken_event.set()
            #files = folders = 0
            #for _, dirnames, filenames in os.walk("/home/pi/programs/images/"):
              # ^ this idiom means "we won't be using this value"
                #files += len(filenames)
                #folders += len(dirnames)
            directory = "/home/pi/programs/images/"
            folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
            if nr < 10:
                newname = folder + "/IMG-000" + str(nr) + ".jpg"
            else:
                if nr < 100:
                    newname = folder + "/IMG-00" + str(nr) + ".jpg"
                else:
                    if nr < 1000:
                        newname = folder + "/IMG-0" + str(nr) + ".jpg"
                    else:
                        newname = folder + "/IMG-" + str(nr) + ".jpg"
            print("this is newname bevore copying: ")
            print(newname)
            shutil.copy("/home/pi/programs/newimage/new.jpg", newname)
            nr += 1


def led_countdown(e): #e is first button pushed
    iteration = 0
    offset = 0 #offset for countdown animation
    offset_idle = 0 #offset for idle animation
    old_color = 0x0
    new_color = 0x1
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
            new_color = DOT_COLORS[iteration]
            for i in range(LED_COUNT):
                if i < offset_idle:
                    color = new_color
                else:
                    color = old_color
                ws.ws2811_led_set(channel, i, color)
                resp = ws.ws2811_render(leds)
            offset_idle += 1
            time.sleep(0.03)
            if offset_idle == LED_COUNT + 1:
                iteration += 1
                offset_idle = 0
                #offsetidle = random.randint(0,23)
                old_color = new_color
            if iteration == len(DOT_COLORS):
                iteration = 0
            



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
    GPIO.setmode(GPIO.BCM)
    button1_pin = 23
    GPIO.setup(button1_pin, GPIO.IN)
    serial = i2c(port=1, address=0x3C)

    # substitute ssd1331(...) or sh1106(...) below if using that device
    device = sh1106(serial)
    with canvas(device) as draw:
            draw.rectangle(device.bounding_box, outline="white", fill="black")
            draw.text((4, 4), "Setup", fill=1)

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
            

    # GPIO callbacks hooks
    GPIO.add_event_detect(button1_pin, GPIO.RISING, callback=but1_callback, bouncetime=300)

    # a process used to run the "long_processing" function in background
    # the first_button_pushed event is passed along
    process_led_count = multiprocessing.Process(name='first_process', target=led_countdown, args=(first_button_pushed,)) #led countdown
    process_camera = multiprocessing.Process(name='camera_process', target=take_photo, args=(animation_finished,))
    process_timer = multiprocessing.Process(name='timer_process', target=timerfunc, args=(animation_finished,))
    process_gallery = multiprocessing.Process(name='gallery_process', target=update_gallery, args=(gallery_update_event,))
    process_oled = multiprocessing.Process(name='oled_process', target=update_oled, args=(oled_update_event,))
    process_backup = multiprocessing.Process(name='backup_process', target=backup_usb)
    
    process_backup.daemon = True
    process_backup.start()
    
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
    
    directory = "/home/pi/programs/images/"
    folder = max([os.path.join(directory,d) for d in os.listdir(directory)], key=os.path.getmtime) #latest created folder
    imglist = []
    def listImages():
        global imglist
        os.chdir(folder)
        imglist = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
        #oldest = files[0]
        #newest = files[-1]
    
    
    def randImg(pics_displayed):
        global imglist
        lastfile = "asldfas"
        listImages()
        collagelist = []
        if imglist == []:
            print("no file found")
            myimage = "/home/pi/programs/countdown/picwait.jpg"
        else:
            if pics_displayed == 4 and os.path.exists(folder + "/collages/"):
                print("displaying collage")
                os.chdir(folder + "/collages/")
                collagelist = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
                myimage = random.choice(collagelist)
                print(myimage)
            else:    
                myimage = random.choice(imglist)
                while (myimage == lastfile) or (myimage == "collages"):
                    #myimage = random.choice(os.listdir(self.imagepath))
                    myimage = random.choice(imglist)
                    
                    print("myimage in while loop")
                    print(myimage)
                lastfile = myimage
                myimage = folder + "/" + myimage
                print(myimage)
        return myimage
    
    def newImg():
        #global imglist
        #listImages()
        #myimage = imglist[-1]
        myimage = "/home/pi/programs/newimage/new.jpg"
        print(myimage)
        return myimage
    
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
    files = folders = 0
    for _, dirnames, filenames in os.walk("/home/pi/programs/images/"):
      # ^ this idiom means "we won't be using this value"
        files += len(filenames)
        folders += len(dirnames)
    print("amount of folders")
    print(folders)
    folder = "/home/pi/programs/images/folder" + str(folders + 1)
    while os.path.exists(folder):
        folders += 1
        folder = "/home/pi/programs/images/folder" + str(folders)
    os.makedirs(folder)
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
        else:
            if first_button_pushed.is_set() and not picwait_displayed == True:
                imagepath = "/home/pi/programs/countdown/picwait.jpg"
                imagechanged = True
                picwait_displayed = True
                print("picwait")
            else:
                if gallery_update_event.is_set():
                    imagepath = randImg(pics_displayed)
                    imagechanged = True
                    pics_displayed += 1
                    if pics_displayed == 5:
                        pics_displayed = 0

                else:
                    imagechanged = False
        if imagechanged == True:
            pilImage = Image.open(imagepath)
            imgWidth, imgHeight = pilImage.size
            if imgWidth > w or imgHeight > h:
                ratio = min(w/imgWidth, h/imgHeight)
                imgWidth = int(imgWidth*ratio)
                imgHeight = int(imgHeight*ratio)
                pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(pilImage)
            imagesprite = canvas.create_image(w/2,h/2,image=image)
            root.update()
            gallery_update_event.clear()
            photo_taken_event.clear()
    

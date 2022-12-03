#https://github.com/davidrazmadzeExtra/RaspberryPi_HTTP_LED/blob/main/http_webserver.py

#https://tutorials-raspberrypi.com/mcp3008-read-out-analog-signals-on-the-raspberry-pi/



#import RPi.GPIO as GPIO
import os
#import cv2
#import math
#import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread, Event, Lock
from time import sleep, time
import subprocess

data_lock = Lock()
event = Event()




host_name = '127.0.0.1'  # IP Address of Raspberry Pi
host_port = 8000

enable = False
threadStarted = False
mythread = Thread()
event = Event()

class MyServer(BaseHTTPRequestHandler):

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def _redirect(self, path):
        self.send_response(303)
        self.send_header('Content-type', 'text/html')
        print(path)
        self.send_header('Location', path)
        self.end_headers()

    def do_GET(self):
        global scale
        html = '''
           <html>
           <title>Fotobooth Webserver</title>
           <body 
            style="width:960px; margin: 20px auto;">
           <h1>Fotobooth Webserver</h1>
           <p>Program to control Fotobooth from webserver </p>
           <div style="width: 800px; float:left;">
           <div style="width: 400px; float:left;">
            <h3>Start Script ".script_to_run.py"</h3>
            <form action="/" method="POST">
                <input type="submit" name="enable" value="enable">
                <input type="submit" name="enable" value="disable">
            </form>
            </div>
           <pre>
           .
           .
           .
           .
           </pre>
           </div>
           <div style="width: 800px; float:left;">
           <h2>Manual Control</h2>
           </div>
           <h3>Seconds to run </h3>
            <form action="/" method="POST">
              <label for="secondsName">Seconds to run motor :</label><br>
              <label for="secondsName">Current Duration:  <b>%s s</b></label><br>
                <input type="submit" name="secondsName" value="1" oninput="this.form.amountRangeSeconds.value=this.value">
                <input type="range" name="amountRangeSeconds" min="1" max="10" value="1" oninput="this.form.secondsName.value=this.value"/>
            </form>
            <div style="width: 800px; float:left;">
            <div style="width: 400px; float:left;">
            <h3>Overall motor speed</h3>
            <form action="/" method="POST">
              <label for="motorspeed">Motor Speed (1 - 100):</label><br>
              <label for="motorspeed">Current Speed:  <b>%s</b></label><br>
                <input type="submit" name="motorspeed" value="1" oninput="this.form.amountRangeMotorspeed.value=this.value">
                <input type="range" name="amountRangeMotorspeed" min="1" max="100" value="1" oninput="this.form.motorspeed.value=this.value"/>
                <!--<input type="number" name="amountInput" min="1" max="100" value="1" oninput="this.form.amountRangeMotorspeed.value=this.value" />-->
            </form>
            </div>
            
            </div>
            <div style="width: 800px; float:left;">
            <div style="width: 400px; float:left;">
            <h3>Individual Motor Control</h3>
            <form action="/" method="POST">

               Motor 1:
               <input type="submit" name="submit" value="M1_left">
               <input type="submit" name="submit" value="M1_right"><br>
                Motor 2:
               <input type="submit" name="submit" value="M2_left">
               <input type="submit" name="submit" value="M2_right"><br>
                Motor 3:
               <input type="submit" name="submit" value="M3_left">
               <input type="submit" name="submit" value="M3_right"><br>
               </form>
            </div>
            <div style="width: 400px; float:left;">
            <h3>Motor Control from Coordination Plane</h3>
           <form action="/" method="POST">
               <input type="submit" name="submit" value="X-Y">
               <input type="submit" name="submit" value="Y">
               <input type="submit" name="submit" value="XY"><br>
               <input type="submit" name="submit" value="X-">
               <input type="submit" name="submit" value="    ">
               <input type="submit" name="submit" value="X"><br>
               <input type="submit" name="submit" value="X-Y-">
               <input type="submit" name="submit" value="Y-">
               <input type="submit" name="submit" value="XY-"><br>
               <br>
               <input type="submit" name="submit" value="rot">
               <input type="submit" name="submit" value="rot-">
           </form>
           </div>
           </div>
           <div style="width: 800px; float:left;">
           <h2>Sensor data</h2>
            <div style="width: 400px; float:left;">
           Encoder M1: %s<br>Encoder M2: %s<br>Encoder M3: %s<br>
           </div>    
           <div style="width: 400px; float:left;">
           Motor Current M1: %s<br>Motor Current M2: %s<br>Motor Current M3: %s<br>
           </div>      
           </div>          
           </body>
           </html>
        ''' % (str(2),str(2),"test","test","test","test","test","test")
        #temp = getTemperature()
        self.do_HEAD()
        #self.wfile.write(html.format(str(scale)).encode("utf-8"))
        #print(html)
        self.wfile.write(html.encode("utf-8"))


    def do_POST(self):
        global scale, speed_m1, speed_m2, speed_m3,motorspeed_manual_fromhtml,seconds_to_run,enable
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode("utf-8")
        print(post_data)
        type_of_data = post_data.split("=")[0]
        post_data = post_data.split("=")[1]
        enable = False
        if type_of_data  == "enable":
            if post_data =="enable":
                enable = True
            elif post_data == "disable":
                enable = False
        self._redirect('/')# Redirect back to the root url
        if enable:
            self.handleThread() #call a thread with given speeds


    def handleThread(self):
        global threadStarted, mythread
        try:
            if threadStarted:
                event.set()
                mythread.join()
        except:
            print("could not join thread")
            pass
        event.clear()
        try:
            mythread = Thread(target=handle_thread, args=(event,enable))
            mythread.start()
            threadStarted = True
        except:
            print("could not start thread")



def handle_thread(event,enable):
    print("handling thread")
    print("value of script enable: ", enable)
    if enable:
        #subprocess.Popen("./script_to_run.py")
        cmd = os.path.join(os.getcwd(), "script_to_run.py")
        os.system('{} {}'.format('python', cmd))
        #os.system("./script_to_run.py")


# # # # # Main # # # # #

if __name__ == '__main__':

    http_server = HTTPServer((host_name, host_port), MyServer)
    print("Server Starts - %s:%s" % (host_name, host_port))
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        http_server.server_close()

    # GPIO.cleanup()
#https://github.com/davidrazmadzeExtra/RaspberryPi_HTTP_LED/blob/main/http_webserver.py

#https://tutorials-raspberrypi.com/mcp3008-read-out-analog-signals-on-the-raspberry-pi/



#import RPi.GPIO as GPIO
import os
#import cv2
#import math
#import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread, Event, Lock
import mimetypes
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

with open("fotobox.html", "r", encoding='utf-8') as f:
    html = f.read()

class MyServer(BaseHTTPRequestHandler):

    def do_HEAD(self):

        # Validate request path, and set type
        print(self.path)
        path = "fotobox.html"
        try:
            ending = self.path.split(".")
            ending = ending[1]
        except:
            ending = None
        path = self.path
        if ending == "html":
            print("serving html")
            type = "text/html"
            path = "." + path
        elif ending == "css":
            print("serving css")
            type = "text/css"
            path = "." + path
        elif ending == "svg":
            print("serving svg")
            type = "image/svg+xml"
            path = "." + path
        else:
            # Wild-card/default
            if not ending == "/":
                print("UNRECONGIZED REQUEST: ", path)
                path = "fotobox.html"


            type = "text/html"

        # Set header with content type
        self.send_response(200)
        self.send_header("Content-type", type)
        self.end_headers()

        # Open the file, read bytes, serve

        with open(path, 'rb') as file:
            self.wfile.write(file.read())  # Send

    def _redirect(self, path):
        self.send_response(303)
        self.send_header('Content-type', 'text/html')
        print(path)
        self.send_header('Location', path)
        self.end_headers()

    def do_GET(self):
        global scale
        global html
        self.do_HEAD()
        ##self.wfile.write(html.format(str(scale)).encode("utf-8"))
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
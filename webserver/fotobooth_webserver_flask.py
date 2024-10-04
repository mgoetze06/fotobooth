from flask import Flask, render_template, request, url_for, redirect, send_from_directory

from flask_socketio import SocketIO, emit

import cgi
from fotobooth_utils import *
from flask import send_file
from glob import glob
from io import BytesIO
from zipfile import ZipFile
import os
import subprocess
import psutil
import datetime
import time


try:
    from gpiozero import CPUTemperature
except:
    pass


app = Flask(__name__)
#            static_url_path='',
#            static_folder='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

photos_temp = 0
stream = None


def readColor():
    try:
        x = readRGBFromFile()
        #print(x)
        newColor = convertTupleToHexString(x)
        #print(newColor)
    except:
        newColor = "#0000000"
        pass
    return newColor

def readImageCount(folder):
    try:
        total_images = str(countFilesInFolder(folder))
    except:
        total_images = ""
        pass
    return total_images

def readCollagesCount(folder):
    try:
        collages = str(countFilesInFolder(folder))
    except:
        collages = ""
        pass
    return collages

def readDataFromFiles():
    global folder
    if folder == None:
        folder = getLatestFolder()
    total_images = readImageCount(folder)
    color = readColor()
    total_collages = readCollagesCount(os.path.join(folder,"collages"))


    disk_usage = getDiskUsage()


    return total_images, color, total_collages

def getCPUValues():
    try:
        cpu = CPUTemperature()
        load = str(round((cpu.temperature/85)*100,2))
        temp = str(round(cpu.temperature,2))
        return temp,load

    except:
        return "0","0"

def getDiskUsage():
    disk = psutil.disk_usage('/')
    disk_free = round((disk.free /2**30),2)
    disk_total = round((disk.total /2**30),2)
    disk_percentage = round((disk_free /disk_total)*100)
    return disk_free,disk_percentage,disk_total

def printRenderingTemplate(total_images, color):
        print("rendering with images: " + total_images + " color: " + color)


def rebootServer():
    print("initiating reboot.")
    try:
        subprocess.call(['sudo','reboot', 'now'])
    except:
        print("reboot failed")
        pass

def shutdownServer():
    print("initiating shutdown.")
    try:
        subprocess.call(['shutdown', '-h', 'now'])
    except:
        print("shutdown failed")
        pass

@socketio.on('createStream')
def createStreamFromFiles():
    global stream
    target = getLatestFolder()
    listImages= glob(os.path.join(target, '*'))
    if len(listImages)>0:
        stream = BytesIO()
        processed = 0
        total = len(listImages)
        with ZipFile(stream, 'w') as zf:
            for file in listImages:
                if not os.path.isdir(file):
                    processed += 1
                    zf.write(file, os.path.basename(file))
                    try:    
                        print("processed",processed)
                        print("total",total)
                        emit('zipfiles', {'processed': processed, 'total': total},broadcast=True)
                    except:
                        print("failed to send zipfiles loading status")
                        pass
                else:
                    total = total -1

        stream.seek(0)
        emit("streamfinished",broadcast=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    global folder
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        files = request.files.getlist("file") 
        for file in files:
            print(file)
            if file.filename == '':
                return redirect(request.url)
            if file: 
                filename = file.filename

                enableCustomCollage(folder)
                folderPath = os.path.join(folder,"customcollage")
                filename = "custom.jpg"
                file.save(os.path.join(folderPath, filename))
                print("file saved",filename)
        return redirect(url_for('on_get'))

@app.route('/download')
def download():
    global stream
    zipName = "FotoboxBilder.zip"
    if stream:
        return send_file(
            stream,
            as_attachment=True,
            download_name=zipName
        )
    else:
        return redirect(url_for('on_get'))

@app.route('/reboot')
def reboot():
    rebootServer()
    return redirect(url_for('on_get'))
@app.route('/shutdown')
def shutdown():
    shutdownServer()
    return redirect(url_for('on_get'))

@app.route('/downloadsingle')
def downloadsingle():
    file = getLatestImage()
    #return send_from_directory('C:\\projects\\fotobooth\\programs\\countdown\\','example_collage.jpg')
    return send_file(file,download_name="test.jpg")

@app.post('/')
def on_post():
    global folder
    if request.method == 'POST':
        data = request.form # a multidict containing POST data
        print(data['color-picker'])
        color = data['color-picker']
        rgbTuple = convertHexToTuple(color)
        writeRGBToFile(rgbTuple)
        if folder:
            clearOldCollages(os.path.join(folder,"collages"))
        total_images, _, _ = readDataFromFiles()
    printRenderingTemplate(total_images,color)
    return redirect(url_for('on_get'))
    #return render_template('index.html', total_images=total_images, color=color)

@app.get('/')
def on_get():
    total_images, color, total_collages = readDataFromFiles()
    printRenderingTemplate(total_images,color)
    return render_template('index.html', total_images=total_images, color=color, total_collages=total_collages)

@socketio.on('settime')
def set_time(data):
    print("Versuche Serverzeit zu setzen: ",data["data"])
    try:
        time = data["data"]
        time.replace(",","")
        subprocess.call(['sudo', 'date', '-s', time])
    except:
        print("Serverzeit setzen fehlgeschlagen.")
        pass

@socketio.on('getvalues')
def get_values(data):
    total_images, color, total_collages = readDataFromFiles()
    printRenderingTemplate(total_images,color)

    emit('values', {'total_images': total_images, 'color': color, 'total_collages': total_collages}, broadcast=True)

    disk_free,disk_percentage,disk_total = getDiskUsage()
    print(disk_free,disk_percentage,disk_total)
    emit('disk', {'disk_free': disk_free, 'disk_percentage': disk_percentage, 'disk_total': disk_total}, broadcast=True)
    time_now = str(datetime.datetime.now().strftime("%H:%M:%S"))
    print(time_now)
    emit('time', {'time_now': time_now}, broadcast=True)

    cpu_temp,cpu_percentage = getCPUValues()
    print(cpu_temp,cpu_percentage)
    emit('cpu', {'cpu_temp': cpu_temp,'cpu_percentage':cpu_percentage}, broadcast=True)

@socketio.on('reboot')
def reboot_server():
    rebootServer()

@socketio.on('shutdown')
def shutdown_server():
    shutdownServer()

def main():
    global folder
    folder = getLatestFolder()
    app.run("0.0.0.0",debug=True)

if __name__ == "__main__":
    main()

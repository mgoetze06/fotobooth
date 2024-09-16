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


app = Flask(__name__)
#            static_url_path='',
#            static_folder='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

photos_temp = 0


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

def readImageCount():
    try:
        total_images = getImagecountFromFile()

    except:
        total_images = ""
        pass
    return total_images

def readCollages():
    try:
        collages = getCollageCountFromFile()

    except:
        collages = ""
        pass
    return collages

def readDataFromFiles():
    total_images = readImageCount()
    color = readColor()
    total_collages = readCollages()
    return total_images, color, total_collages


def printRenderingTemplate(total_images, color):
        print("rendering with images: " + total_images + " color: " + color)

@app.route('/download')
def download():
    target = getLatestFolder()
    listImages= glob(os.path.join(target, '*'))
    if len(listImages)>0:
        zipName = "FotoboxBilder.zip"
        stream = BytesIO()

        with ZipFile(stream, 'w') as zf:
            for file in listImages:
                zf.write(file, os.path.basename(file))
        stream.seek(0)

        return send_file(
            stream,
            as_attachment=True,
            download_name=zipName
        )
    else:
        return redirect(url_for('on_get'))

@app.route('/reboot')
def reboot():
    print("initiating reboot.")
    try:
        subprocess.call(['sudo','reboot', 'now'])
    except:
        print("reboot failed")
        pass

    return redirect(url_for('on_get'))
@app.route('/shutdown')
def shutdown():
    print("initiating shutdown.")
    try:
        subprocess.call(['shutdown', '-h', 'now'])
    except:
        print("shutdown failed")
        pass
    return redirect(url_for('on_get'))

@app.route('/downloadsingle')
def downloadsingle():
    file = getLatestImage()
    #return send_from_directory('C:\\projects\\fotobooth\\programs\\countdown\\','example_collage.jpg')
    return send_file(file,download_name="test.jpg")

@app.post('/')
def on_post():
    if request.method == 'POST':
        data = request.form # a multidict containing POST data
        print(data['color-picker'])
        color = data['color-picker']
        rgbTuple = convertHexToTuple(color)
        writeRGBToFile(rgbTuple)
        total_images, _, _ = readDataFromFiles()
    printRenderingTemplate(total_images,color)
    return redirect(url_for('on_get'))
    #return render_template('index.html', total_images=total_images, color=color)

@app.get('/')
def on_get():
    total_images, color, total_collages = readDataFromFiles()
    printRenderingTemplate(total_images,color)
    return render_template('index.html', total_images=total_images, color=color, total_collages=total_collages)


@socketio.on('getvalues')
def get_values(data):
    global photos_temp
    photos_temp = photos_temp + 1
    emit('values', {'data': photos_temp}, broadcast=True)


def main():
    app.run("0.0.0.0",debug=True)


if __name__ == "__main__":
    main()

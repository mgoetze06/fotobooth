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
import tempfile
import shutil
import mimetypes


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
zip_chunk_dir = None
zip_chunk_files = []


def cleanup_zip_chunks():
    global zip_chunk_dir, zip_chunk_files
    if zip_chunk_dir and os.path.exists(zip_chunk_dir):
        try:
            shutil.rmtree(zip_chunk_dir)
        except Exception:
            pass
    zip_chunk_dir = None
    zip_chunk_files = []


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

def getCountdown():
    countDownActiveBool = getCountdownFromFile()
    if countDownActiveBool:
        return "Aktiv"
    return "Inaktiv"

@socketio.on('createStream')
def createStreamFromFiles():
    global stream, zip_chunk_dir, zip_chunk_files
    target = getLatestFolder()
    listImages = [path for path in glob(os.path.join(target, '*')) if not os.path.isdir(path)]
    if len(listImages) == 0:
        emit('zipfileserror', {'error': 'No files found for download'})
        return

    cleanup_zip_chunks()
    zip_chunk_dir = tempfile.mkdtemp(prefix='fotobooth_zip_')
    zip_chunk_files = []

    max_files_per_chunk = 50
    max_bytes_per_chunk = 80 * 1024 * 1024
    current_chunk = []
    current_size = 0
    total = len(listImages)
    processed = 0
    chunks = []

    for file in sorted(listImages):
        file_size = os.path.getsize(file)
        if current_chunk and (len(current_chunk) >= max_files_per_chunk or current_size + file_size > max_bytes_per_chunk):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(file)
        current_size += file_size
    if current_chunk:
        chunks.append(current_chunk)

    for chunk_index, chunk_files in enumerate(chunks, start=1):
        zip_name = f'FotoboxBilder_part{chunk_index:02d}.zip'
        zip_path = os.path.join(zip_chunk_dir, zip_name)
        with ZipFile(zip_path, 'w') as zf:
            for file in chunk_files:
                zf.write(file, os.path.basename(file))
                processed += 1
                try:
                    emit('zipfiles', {'processed': processed, 'total': total}, broadcast=True)
                except Exception:
                    pass

        zip_chunk_files.append({
            'index': chunk_index - 1,
            'path': zip_path,
            'name': zip_name,
            'count': len(chunk_files),
            'size': os.path.getsize(zip_path)
        })

    emit('zipchunksready', {
        'chunks': [
            {
                'index': info['index'],
                'name': info['name'],
                'count': info['count'],
                'size': info['size']
            }
            for info in zip_chunk_files
        ]
    }, broadcast=True)

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

@app.route('/lastimage')
def last_image():
    return render_template('last_image.html', image_url=url_for('last_image_file'))

@app.route('/image/last')
def last_image_file():
    file_path = getLatestImage()
    if not os.path.exists(file_path):
        return redirect(url_for('on_get'))
    mime_type, _ = mimetypes.guess_type(file_path)
    return send_file(file_path, mimetype=mime_type or 'application/octet-stream')

@app.route('/image/last/meta')
def last_image_meta():
    file_path = getLatestImage()
    if not os.path.exists(file_path):
        return {'exists': False}
    return {
        'exists': True,
        'filename': os.path.basename(file_path),
        'modified': os.path.getmtime(file_path)
    }

@app.route('/downloadchunk/<int:chunk_index>')
def download_chunk(chunk_index):
    global zip_chunk_files
    if chunk_index < 0 or chunk_index >= len(zip_chunk_files):
        return redirect(url_for('on_get'))
    info = zip_chunk_files[chunk_index]
    return send_file(
        info['path'],
        as_attachment=True,
        download_name=info['name']
    )

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
    countdown_active = getCountdown()
    countDownSleepTimeSeconds = getSleepTimeSecondsFromFile()
    printRenderingTemplate(total_images,color)
    return render_template('index.html', total_images=total_images, color=color, total_collages=total_collages, countdown_active = countdown_active,countDownSleepTimeSeconds=countDownSleepTimeSeconds)

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


@socketio.on('toggleCountdown')
def toggle_countdown(data):
    if getCountdownFromFile():
        deactivateCountdownBeforeTakingPicture()
    else:
        activateCountdownBeforeTakingPicture()

    countdown = getCountdown()
    emit("countdown",{'countdown': countdown})

@socketio.on('setCountDownSleepTimeSeconds')
def set_CountDownSleepTimeSeconds(data):
    seconds = getSleepTimeSecondsFromFile()

    emit("CountDownSleepTimeSeconds",{'CountDownSleepTimeSeconds': seconds})

@socketio.on('increaseSleepTimeSeconds')
def server_increaseSleepTimeSeconds(data):
    increaseSleepTimeSeconds()
    set_CountDownSleepTimeSeconds("")
@socketio.on('decreaseSleepTimeSeconds')
def server_decreaseSleepTimeSeconds(data):
    decreaseSleepTimeSeconds()
    set_CountDownSleepTimeSeconds("")

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

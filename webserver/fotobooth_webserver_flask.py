from flask import Flask, render_template, request, url_for, redirect, send_from_directory
import cgi
from fotobooth_utils import convertHexToTuple, writeRGBToFile,readRGBFromFile,convertTupleToHexString,getImagecountFromFile
from flask import send_file
from glob import glob
from io import BytesIO
from zipfile import ZipFile
import os


app = Flask(__name__)
#            static_url_path='',
#            static_folder='/static')
app.config['SECRET_KEY'] = 'secret!'

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

def readDataFromFiles():
    total_images = readImageCount()
    color = readColor()
    return total_images, color


def printRenderingTemplate(total_images, color):
        print("rendering with images: " + total_images + " color: " + color)

@app.route('/download')
def download():
    target = 'C:\\projects\\fotobooth\\programs\\countdown'
    zipName = "FotoboxBilder.zip"
    stream = BytesIO()
    with ZipFile(stream, 'w') as zf:
        for file in glob(os.path.join(target, '*.jpg')):
            zf.write(file, os.path.basename(file))
    stream.seek(0)

    return send_file(
        stream,
        as_attachment=True,
        download_name=zipName
    )


@app.route('/downloadsingle')
def downloadsingle():
    file = 'C:\\projects\\fotobooth\\programs\\countdown\\example_collage.jpg'
    #return send_from_directory('C:\\projects\\fotobooth\\programs\\countdown\\','example_collage.jpg')
    return send_file('C:\\projects\\fotobooth\\programs\\countdown\\example_collage.jpg',download_name="test.jpg")

@app.post('/')
def on_post():
    if request.method == 'POST':
        data = request.form # a multidict containing POST data
        print(data['color-picker'])
        color = data['color-picker']
        rgbTuple = convertHexToTuple(color)
        writeRGBToFile(rgbTuple)
        total_images, _ = readDataFromFiles()
    printRenderingTemplate(total_images,color)
    return redirect(url_for('on_get'))
    #return render_template('index.html', total_images=total_images, color=color)

@app.get('/')
def on_get():
    total_images, color = readDataFromFiles()
    printRenderingTemplate(total_images,color)
    return render_template('index.html', total_images=total_images, color=color)


def main():

    app.run("0.0.0.0",debug=True)


if __name__ == "__main__":
    main()

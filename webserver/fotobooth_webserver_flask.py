from flask import Flask, render_template, request, url_for, redirect
import cgi
from fotobooth_utils import convertHexToTuple, writeRGBToFile,readRGBFromFile,convertTupleToHexString,getImagecountFromFile

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

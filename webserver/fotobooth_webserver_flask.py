from flask import Flask, render_template, request, url_for
import cgi
from fotobooth_utils import convertHexToTuple, writeRGBToFile,readRGBFromFile,convertTupleToHexString,getImagecountFromFile

app = Flask(__name__)
#            static_url_path='',
#            static_folder='/static')
app.config['SECRET_KEY'] = 'secret!'


@app.route('/', methods = ['GET', 'POST'])
def serveNormal():

    try:
        total_images = getImagecountFromFile()

    except:
        total_images = ""
        pass

    try:
        x = readRGBFromFile()
        print(x)
        newColor = convertTupleToHexString(x)
        print(newColor)

    except:
        total_images = ""
        pass



    if request.method == 'GET':
        print("get")
    if request.method == 'POST':
        print("post")
        data = request.form # a multidict containing POST data
        print(data['color-picker'])
        color = data['color-picker']
        rgbTuple = convertHexToTuple(color)
        writeRGBToFile(rgbTuple)
        try:
            x = readRGBFromFile()
            print(x)
            newColor = convertTupleToHexString(x)
            print(newColor)

        except:
            total_images = ""
            
            
    print("rendering with images: " + total_images + " color: " + newColor)
    return render_template('index.html', total_images=total_images, color=newColor)


def main():

    app.run(debug=True)


if __name__ == "__main__":
    main()

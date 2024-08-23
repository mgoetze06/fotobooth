from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi
from fotobooth_utils import convertHexToTuple, writeRGBToFile,readRGBFromFile,convertTupleToHexString,getImagecountFromFile




class webpage():
    def __init__(self,filepath) -> None:
        try:
            self.basefile = open(filepath).read()
        except:
            pass
        pass
    def getPage(self):
        self.updatePageFromFiles()
        return bytes(self.file, 'utf-8')
                
    
    def updatePageFromFiles(self):
        # update color
        html_text = "{{default_color}}"
        
        x = readRGBFromFile()
        print(x)
        newColor = convertTupleToHexString(x)
        print(newColor)
        self.file = self.basefile.replace(html_text, newColor)


        # update photos taken
        html_text = "{{total_images}}"
        amountOfImages = getImagecountFromFile()
        self.file = self.file.replace(html_text, amountOfImages)




class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            print("pfad webserver: ",self.path)
            self.path = '/index.html'
        try:
            #file_to_open = open(self.path[1:]).read()
            w = webpage(self.path[1:])
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(w.getPage())
        except:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'404 - Not Found')
    def do_POST(self):
            if self.path == '/':
                self.path = '/index.html'
            # Parse the form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': self.headers['Content-Type'],
                        }
            )

            # Get the form values
            first_name = form.getvalue("first_name")
            last_name = form.getvalue("last_name")
            print(last_name)
            color = form.getvalue("color-picker")
            print(color)
            rgbTuple = convertHexToTuple(color)

            writeRGBToFile(rgbTuple)


            w = webpage(self.path[1:])
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(w.getPage())

def main():
    httpd = HTTPServer(('', 8000), SimpleHTTPRequestHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi

def writeToFile(rgb_tuple):
    f = open("color.txt", "w") 
    rgb = "RGB"
    for i in range(3):
         f.write(rgb[i])
         f.write(str(rgb_tuple[i]))
         f.write('\n')
    f.close()


def convertHexToTuple(hex):
     #    value is: #FF0000
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        try:
            file_to_open = open(self.path[1:]).read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))
        except:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'404 - Not Found')
    def do_POST(self):
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

            writeToFile(rgbTuple)

            # bytes_obj = bytes.fromhex(color)
            # result_string = bytes_obj.decode('utf-8')
            # print(result_string)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'Farbe wurde aktualisiert! ' + color.encode())

httpd = HTTPServer(('', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
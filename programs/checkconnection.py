try:
    import httplib
except:
    import http.client as httplib
import dropbox,os
def have_internet():
    conn = httplib.HTTPConnection("www.google.com", timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False
    
    
    
if have_internet():
    print("connected")
    dropbox_access_token= "o2Q0QpxTIowAAAAAAAAAAYmvft-NxQmKRDhaf1Ba4w2wxJlEbC42IPX0XuQTjiJQ"
    #dropbox_access_token= "sl.AvR0AT-pXGC_-aeAPS_JFdkUa5_lvpOU_1lFX3pAWgbxPcgcCuMdAmfjlzlJ7lODi-WmJguu1nVGLNnuWuKXCWEnLdRZGaarXyzJrNe5cc_ARZKPVKpLLE0mH9fn8B1fqpPxQ2Y"    #Enter your own access token
    #dropbox_access_token= "m5xjb0fj87lqpnq"
    client = dropbox.Dropbox(dropbox_access_token)
    print("[SUCCESS] dropbox account linked")
    for root, dirs, files in os.walk("/home/pi/programs/collages/"):

        for filename in files:

            # construct the full local path
            #local_path = os.path.join(root, filename)
            computer_path = os.path.join(root, filename)
            # construct the full Dropbox path
            #relative_path = os.path.relpath(local_path, "/home/pi/programs/collages/")
            #dropbox_path = os.path.join("/collages/", relative_path)
            dropbox_path= "/collages/" + filename
            
            
            client.files_upload(open(computer_path, "rb").read(), dropbox_path)
            print("[UPLOADED] {}".format(computer_path))
    #dropbox_path= "/collages/Collage1.jpg"
    #computer_path="/home/pi/programs/collages/Collage1.jpg"
    
    
    #client.files_upload(open(computer_path, "rb").read(), dropbox_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
else:
    print("not connected")
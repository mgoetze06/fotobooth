#import pyudev
import shutil,os,re
from distutils.dir_util import copy_tree
import datetime
#print(pyudev.__version__)
#context = pyudev.Context()
#for device in context.list_devices(subsystem='block'):
#    print(device)

#monitor = pyudev.Monitor.from_netlink(context)
#monitor.filter_by('block')

#polling status (blocking any loop)
#for device in iter(monitor.poll, None):
#    if 'ID_FS_TYPE' in device:
#        print('{0} partition {1}'.format(device.action, device.get('ID_FS_LABEL')))
#        print(device.get('ID_FS_TYPE'))
#        print(device.action)
#        if device.action == 'add':
#            print("usb added")
#            print("my name is " + device.get('ID_FS_LABEL'))
#            
#            path = "/dev/" + device.get('ID_FS_LABEL')
            
devices = os.popen('sudo blkid').readlines()
usbs = []
for u in devices:
    loc = [u.split(':')[0]]
    if '/dev/sd' not in loc[0]: 
        continue # skip 
    loc+=re.findall(r'"[^"]+"',u)
    columns = ['loc']+re.findall(r'\b(\w+)=',u)
                
    usbs.append(dict(zip(columns,loc)))

for u in usbs:
    print ('Device %(LABEL)s is located at $(loc)s with UUID of $(UUID)s'%u )
    name = "%(LABEL)s" %u
    name = name[:-1]
    name = name[1:]
    print(name)
os.system('sudo mount $(loc)s /myusb'%u)
            
dest = "/media/pi/"+ name +"/NeueTestFile.jpg"
shutil.copy("/home/pi/programs/newimage/new.jpg", dest)
dest = "/media/pi/"+ name +"/programs_backup"
copy_tree("/home/pi/programs", dest)

today = str(datetime.datetime.now())
with open('/home/pi/programs/log_backup.txt', 'w') as f:
    f.write(today)
    f.close()


quit()
#def log_event(action, device):
#    if 'ID_FS_TYPE' in device:
#        with open('filesystems.log', 'a+') as stream:
#            print('{0} - {1}'.format(action, device.get('ID_FS_LABEL')), file=stream)
#            
#observer = pyudev.MonitorObserver(monitor, callback=log_event)
#observer.start()
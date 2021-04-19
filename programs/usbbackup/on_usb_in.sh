#!/bin/sh
#   sudo killall python
sudo pkill -f fotobooth.py
/usr/bin/python3 /home/pi/programs/usbbackup/backup_usb.py

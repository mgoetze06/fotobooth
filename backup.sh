#!/bin/sh
sudo killall python
python3 /home/pi/programs/usbbackup/backup_usb.py
sudo killall python
sudo python3 /home/pi/programs/fotobooth.py
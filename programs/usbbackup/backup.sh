#!/bin/sh
sudo killall python
python /home/pi/programs/usbbackup/backup_usb.py
sudo killall python
sudo python /home/pi/programs/fotobooth.py
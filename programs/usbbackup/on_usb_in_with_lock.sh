#!/bin/sh
LOCK=/tmp/lockfile_for_plug_usb
if [ -f $LOCK ]
then
   exit 1
else
   touch $LOCK;
   # the actual command to run upon USB plug in
#   sudo killall python
   sudo pkill -f fotobooth.py
   /usr/bin/python3 /home/pi/programs/usbbackup/backup_usb.py

fi

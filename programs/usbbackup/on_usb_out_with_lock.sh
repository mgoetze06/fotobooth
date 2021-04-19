#!/bin/sh

LOCK=/tmp/lockfile_for_plug_usb/bin/
sudo rm -f /tmp/lockfile_for_plug_usb

sudo python /home/pi/programs/fotobooth.py

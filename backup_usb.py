#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SOURCE = Path('/home/pi/programs')
DEFAULT_DEST_NAME = 'programs_backup'
DEFAULT_DELAY = 60


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
    return result.stdout.strip()


def parse_lsblk():
    output = run_command(['lsblk', '-P', '-o', 'NAME,TRAN,TYPE,MOUNTPOINT,UUID,LABEL'])
    devices = []
    for line in output.splitlines():
        fields = dict(re.findall(r'(\w+)="([^"]*)"', line))
        devices.append(fields)
    return devices


def is_writable_mount(mountpoint):
    mount_path = Path(mountpoint)
    if not mount_path.is_dir() or not os.access(mount_path, os.W_OK):
        return False
    try:
        test_file = mount_path / '.usb_write_test'
        with open(test_file, 'w') as f:
            f.write('ok')
        test_file.unlink()
        return True
    except OSError:
        return False


def find_usb_storage():
    devices = parse_lsblk()
    candidates = []

    for dev in devices:
        if dev.get('TRAN') != 'usb':
            continue
        if dev.get('TYPE') not in ('part', 'disk'):
            continue
        mountpoint = dev.get('MOUNTPOINT')
        if not mountpoint:
            continue
        if is_writable_mount(mountpoint):
            candidates.append({'device': dev, 'mountpoint': Path(mountpoint)})

    if candidates:
        return candidates

    if os.geteuid() == 0:
        for dev in devices:
            if dev.get('TRAN') != 'usb' or dev.get('TYPE') != 'part':
                continue
            if dev.get('MOUNTPOINT'):
                continue
            name = dev.get('NAME')
            if not name:
                continue
            mount_dir = Path('/mnt') / f'usb_{name}'
            mount_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(['mount', f'/dev/{name}', str(mount_dir)], check=True)
            except subprocess.CalledProcessError:
                continue
            if is_writable_mount(mount_dir):
                candidates.append({'device': dev, 'mountpoint': mount_dir})
    return candidates


def copy_directory(source, destination):
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f'Source folder does not exist: {source_path}')
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(source_path):
        relative_root = Path(root).relative_to(source_path)
        target_root = destination_path / relative_root
        target_root.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            source_file = Path(root) / file_name
            target_file = target_root / file_name
            shutil.copy2(source_file, target_file)


def main():
    parser = argparse.ArgumentParser(
        description='Backup a folder to a writable USB storage device on Raspberry Pi.'
    )
    parser.add_argument(
        '--source', default=str(DEFAULT_SOURCE),
        help='Internal source folder to copy from (default: /home/pi/programs)'
    )
    parser.add_argument(
        '--dest-name', default=DEFAULT_DEST_NAME,
        help='Folder name to create on the USB device (default: programs_backup)'
    )
    parser.add_argument(
        '--delay', type=int, default=DEFAULT_DELAY,
        help='Seconds to wait before starting the copy (default: 60)'
    )
    parser.add_argument(
        '--verbose', action='store_true', help='Print debug information'
    )
    args = parser.parse_args()

    if args.delay > 0:
        print(f'Waiting {args.delay} seconds before backup...')
        for remaining in range(args.delay, 0, -1):
            print(f'  {remaining} seconds remaining', end='\r', flush=True)
            time.sleep(1)
        print(' ' * 60, end='\r')

    candidates = find_usb_storage()
    if not candidates:
        print('No writable USB storage device detected. Please insert and mount a writable USB drive.')
        sys.exit(1)

    usb = candidates[0]
    device = usb['device']
    mountpoint = usb['mountpoint']
    label = device.get('LABEL') or device.get('NAME')
    print(f'Using USB device {device.get("NAME")} label={label} mounted at {mountpoint}')

    destination = mountpoint / args.dest_name
    if args.verbose:
        print(f'Copying from {args.source} to {destination}')

    copy_directory(args.source, destination)

    log_path = Path('/home/pi/programs/log_backup.txt')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as log_file:
        log_file.write(f'Backup completed to {destination} at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    print(f'Backup finished successfully to {destination}')


if __name__ == '__main__':
    main()

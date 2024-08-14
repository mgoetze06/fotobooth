import os,time
from datetime import datetime
def creation_date(path_to_file):
#"""
#Try to get the date that a file was created, falling back to when it was
#last modified if that isn't possible.
#See http://stackoverflow.com/a/39501288/1709587 for explanation.
#"""
    stat = os.stat(path_to_file)
    try:
        return stat.st_birthtime
    except AttributeError:
        # We're probably on Linux. No easy way to get creation dates here,
        # so we'll settle for when its content was last modified.
        #return time.ctime(stat.st_mtime)
        return datetime.fromtimestamp(stat.st_mtime)


def checkAndCreateFolder(parent_path,new_folder):
    if not parent_path.endswith("/"):
        parent_path = parent_path + "/"
    folder_to_check = parent_path + new_folder
    if os.path.exists(folder_to_check):
        print(parent_path + " contains " + new_folder +" already.")
    else:
        print(parent_path + " does not contain " + new_folder)
        os.makedirs(folder_to_check)
        print("directory " + new_folder + " created.")
    print(os.listdir(folder_to_check))
    folders = [name for name in os.listdir(folder_to_check) if os.path.isdir(os.path.join(folder_to_check, name))]


    folders = len(folders)

    print("amount of folders")
    print(folders)
    folder = "/home/pi/programs/images/folder" + str(folders)
    print("last folder: ", folder)
    folderdate = creation_date(folder)
    print("folder creation time: ", folderdate)
    currenttime = datetime.fromtimestamp(time.time())
    print("current time: ", currenttime)
    timediff_minutes = abs((folderdate - currenttime).total_seconds()/60) #timediff in minutes
    print(timediff_minutes)
    if(timediff_minutes > 60*12):
        folder = "/home/pi/programs/images/folder" + str(folders + 1)
        print("old folder. need to create new one: ", folder)
    else:
        print("folder is not old enough. reuse folder: ", folder)
    while os.path.exists(folder):
        folders += 1
        folder = "/home/pi/programs/images/folder" + str(folders)
    os.makedirs(folder)
    return folder

folder = checkAndCreateFolder("/home/pi/programs","images")
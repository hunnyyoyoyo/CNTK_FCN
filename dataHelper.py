import sys
import os
import zipfile
import fnmatch

data_path = os.path.join("data/M4")
zip_path = os.path.join("data-zip")

def hydrate(zip_path, dest_path):
    print("Start unzipping data files in {0} to {1}".format(zip_path,dest_path))
    if (os.path.exists(zip_path) == False):
        print("The source folder {0} doesn't exist, so quitting".format(zip_path))
        quit()

    zipfile_count = len(fnmatch.filter(os.listdir(zip_path), '*.zip'))
    if (zipfile_count == 0):
        print("No zip (.zip) files in {0}, so quitting ".format(zip_path))

    print("zip file count:%s" % zipfile_count)

    if (os.path.exists(dest_path) == False):
        print("Destination folder {0} doesn't exist, creating it".format(dest_path))
        os.makedirs(dest_path)

        # Extract all zip files from zip_path to dest_path
        print("Start unzipping files to {0}".format(dest_path))
        for item in os.listdir(zip_path): # loop through items in dir
            if item.endswith(".zip"): # check for ".zip" extension
                print("   unzipping {0} ...".format(item))
                file_name = os.path.join(zip_path,item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall(dest_path) # extract file to dir
                zip_ref.close() # close file
    else:
        print("data folder already populated")

    print("Complete: Files have been unzipped to {0}".format(dest_path))
    
hydrate(zip_path, data_path)
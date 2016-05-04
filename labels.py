# import the necessary packages
import numpy as np
import argparse
import pandas as pd
import os.path
from xlsxwriter.utility import xl_rowcol_to_cell, xl_cell_to_rowcol
from os.path import join
from glob import glob

def get_file_list(image_directory = None, template_image = "./data/kernel.PNG"):
# construct the argument parser and parse the arguments
    global NUM_COLS
    global NUM_ROWS
    global kernel_file
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = False, help = "Path to the image", default = None)
    ap.add_argument("-d", "--dir", required = False, help = "Path to the dir of images", default = image_directory)
    ap.add_argument("-t", "--template", required = False, help = "Path to the example cropped plate image you want to use.", default = template_image)
    ap.add_argument("-s", "--size", required = False, help = "Size of your plate", default = "384")
    args = vars(ap.parse_args())
    if int(args["size"]) == 384:
        NUM_COLS = 24
        NUM_ROWS = 16
    else:
        print args["size"]
        raise Exception("Unsupported size specified")

    kernel_file = str(args['template'])
    if check_is_image(kernel_file)==False:
        raise Exception("Template file "+str(kernel_file)+" not valid image file.")

    if args["dir"] != None:
        files = []
        for ext in ('*.gif', '*.png', '*.jpg', '*.PNG', '*.JPG'):
            files.extend(glob(join(args["dir"], ext)))
    else:
        files = [args["image"]]
    for file in files:
        if check_is_image(file)==False:
            raise Exception("File " +str(file)+" not a valide image file.")
    print "Going to process:"
    print files
    return files

if __name__ == "__main__":
    files = ['./data/labels/1GS1BT.csv', './data/labels/1GS2BT.csv', './data/labels/1GS3BT.csv', './data/labels/1GS4BT.csv']
    file_order = [0, 1,2,3] # which position each plate corresponds to on the master plate
    plates_data = [pd.read_csv(i,header=1) for i in files]
    for i, plate in enumerate(plates_data):
        #plate.rename(columns={'ORF': 'ORF_original', 'SGD': 'SGD_original', 'Controls added ': 'Control'}, inplace=True)
#delete irrelevant columns/rows
        plate.drop(plate.columns[[range(6,len(plate.columns))]],axis=1, inplace = True)
        plate.drop(plate.index[[range(96,plate.shape[0])]],axis=0, inplace = True)
        plate.columns= ['ORF_original', 'SGD_original', 'Plate', 'Well', 'to_delete', 'Control']
        plate.drop('to_delete', axis=1, inplace=True)
        plate['SGD'] = np.where(pd.isnull(plate['Control']), plate['SGD_original'], plate['Control'])
        plate['ORF'] = np.where(pd.isnull(plate['Control']), plate['ORF_original'], plate['Control'])
        plate['Column'], plate['Row'] = zip(*plate['Well'].apply(xl_cell_to_rowcol))
        print plate
        plate.to_csv('./results_numbers/base_' + str(os.path.basename(files[i])))
        #transform each row/column to the correct row/column on the master plate
        plate['Row'] *= 2
        plate['Column'] *= 2
        if (i>1):
            plate['Row'] += 1
        if (i%2 == 1):
            plate['Column'] += 1
        print plate

        plate.set_index(['Row', 'Column'], inplace=True)
        print plate
    master_plate = pd.concat(plates_data)
    print master_plate.sort_index()



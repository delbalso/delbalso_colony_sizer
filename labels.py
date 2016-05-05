# import the necessary packages
import numpy as np
import argparse
import pandas as pd
import os.path
from xlsxwriter.utility import xl_rowcol_to_cell, xl_cell_to_rowcol
from os.path import join
import imghdr
from glob import glob
import dbcolonysizer


replicate_order = {0:[0, 1,2,3], 1:[1,0,3,2], 2:[2,3,0,1],3:[3,2,1,0]} # which position each plate corresponds to on the master plate
def process_experiments(path):
    pinnings = {}
    replicates = {}
    experiments = {}
    days = {}
    for root, dirs, files in os.walk(path):
        # keep only images
        files = [file for file in files if imghdr.what(os.path.join(root,file))!=None]
        if len(files)==0:
            continue
        path = root
        path, pinning = os.path.split(path)
        path, replicate = os.path.split(path)
        path, experiment = os.path.split(path)
        pinnings[pinning]=1
        replicates[replicate]=1
        experiments[experiment]=1
        for day in files:
            days[day] = 1
        #print "Experiment: "+str(experiment)+". Replicate: "+str(replicate)+". Pinning: "+str(pinning)+". Days: "+str(files)

    print "Experiments: " + str(experiments.keys())
    print "Replicates: " + str(replicates.keys())
    print "Pinnings: " + str(pinnings.keys())
    print "Days: " + str(days.keys())
    iterables = [experiments.keys(), replicates.keys(), pinnings.keys(), [os.path.splitext(day)[0] for day in days.keys()]]
    index = pd.MultiIndex.from_product(iterables, names=['Experiments', 'Replicates', 'Pinnings', 'Days'])
    a = pd.DataFrame(columns = index)
    print a

    # process files and populate main table
    for root, dirs, files in os.walk(path):
        # keep only images
        files = [file for file in files if imghdr.what(os.path.join(root,file))!=None]
        if len(files)==0:
            continue
        path = root
        path, pinning = os.path.split(path)
        path, replicate = os.path.split(path)
        path, experiment = os.path.split(path)
    raise

#process a single day's data, this is for a given experiment, replica, and pinning
def process_day(filename):
    day_data = dbcolonysizer.process_files(filename)
    print day_data
    return day_data

# Returns a list of positions and genes in that position. The index of this dataframe is the position
def replicate_to_gene_list(replicate):
    files = ['./data/labels/1GS1BT.csv', './data/labels/1GS2BT.csv', './data/labels/1GS3BT.csv', './data/labels/1GS4BT.csv']
    file_order = replicate_order[replicate]
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
        #print plate
        plate.to_csv('./results_numbers/base_' + str(os.path.basename(files[i])))
        #transform each row/column to the correct row/column on the master plate
        plate['Row'] *= 2
        plate['Column'] *= 2
        print "Mapping plate "+str(i)+" to position " + str(file_order[i])
        if file_order[i] == 0:
            pass #do nothing
        elif file_order[i] == 1:
            plate['Column'] += 1
        elif file_order[i] == 2:
            plate['Row'] += 1
        elif file_order[i] == 3:
            plate['Row'] += 1
            plate['Column'] += 1

        plate.set_index(['Row', 'Column'], inplace=True)
    master_plate = pd.concat(plates_data).sort_index()
    return master_plate

if __name__ == "__main__":
    dbcolonysizer.initialize()
    print replicate_to_gene_list(1)
    raise
    process_experiments('./data/experiments')
    #process_pinning('./data/experiments/experiment1/replicate1/pin1')
    #process_day('./data/experiments/experiment1/replicate1/pin1/day1.JPG')
    raise



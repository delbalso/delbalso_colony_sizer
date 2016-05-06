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
RESULTS_FOLDER = './results_numbers/'
NUM_COLS = 24
NUM_ROWS = 16
replicate_order = {
    'replicate1': [
        0, 1, 2, 3], 'replicate2': [
            1, 0, 3, 2], 'replicate3': [
                2, 3, 0, 1], 'replicate4': [
                    3, 2, 1, 0]}  # which position each plate corresponds to on the master plate

""" Process a batch of experiments where path is a dir of experiment directories conforming to a experiment>treatment>replicate>pinning>day hierarchy where day is the leaf and is an image file."""


def process_batch(path):
    pinnings = {}
    replicates = {}
    experiments = {}
    treatments = {}
    days = {}
    data_to_process = []
    for root, dirs, files in os.walk(path):
        # keep only images
        files = [
            file for file in files if imghdr.what(
                os.path.join(
                    root, file)) != None]
        if len(files) == 0:
            continue
        path = root
        path, pinning = os.path.split(path)
        path, replicate = os.path.split(path)
        path, treatment = os.path.split(path)
        path, experiment = os.path.split(path)
        pinnings[pinning] = 1
        replicates[replicate] = 1
        treatments[treatment] = 1
        experiments[experiment] = 1
        for day in files:
            filepath = os.path.join(root, day)
            days[day] = 1
            data_to_process.append(
                (filepath, experiment, treatment, replicate, pinning, day))
        # print "Experiment: "+str(experiment)+". Replicate:
        # "+str(replicate)+". Pinning: "+str(pinning)+". Days: "+str(files)
    assert len(
        data_to_process) > 0, "No valid files found to process. Check the directory."
    print "Processing {0} files".format(len(data_to_process))
    print "Experiments: " + str(experiments.keys())
    print "Treatments: " + str(treatments.keys())
    print "Replicates: " + str(replicates.keys())
    print "Pinnings: " + str(pinnings.keys())
    print "Days: " + str(days.keys())
    print "Data to process: " + str(data_to_process)
    c_iterables = [experiments.keys(), treatments.keys(), replicates.keys(
    ), pinnings.keys(), [os.path.splitext(day)[0] for day in days.keys()]]
    c_index = pd.MultiIndex.from_product(
        c_iterables,
        names=[
            'Experiments',
            'Treatments',
            'Replicates',
            'Pinnings',
            'Days'])
    # get a list of all genes in all experiments
    gene_list = {}  # key is replicate, values are dataframes of row/col & genes
    gene_dict = {}  # keys are genes. used for deduping
    for replicate in replicates:
        gene_list[replicate] = get_gene_list(replicate)
        for gene in gene_list[replicate]['SGD']:
            gene_dict[gene] = 1
    genes = gene_dict.keys()

    master_table_not_defined = True
    master_table = None

    # process files and populate main table
    for filepath, experiment, treatment, replicate, pinning, day in data_to_process:

        day_data = process_day(filepath)
        day_data.columns = ['single_column']
        day_data = day_data.join(gene_list[replicate])
        day_data.set_index('ORF', inplace=True)

        # Make new index that has different levels
        day_index = pd.MultiIndex.from_tuples(
            [
                (experiment, treatment, replicate, pinning, os.path.splitext(day)[0])], names=[
                'Experiments', 'Treatments', 'Replicates', 'Pinnings', 'Days'])
        day_data = day_data['single_column']
        day_data = pd.DataFrame(
            day_data.values,
            columns=day_index,
            index=day_data.index)
        groups = day_data.groupby(level=0)  # dedupe repeated indices
        day_data = groups.last()
        if master_table_not_defined:
            master_table = day_data
            master_table_not_defined = False
        else:
            master_table = master_table.join(day_data, how='outer')
        print master_table.shape
    print master_table
    master_table.to_csv(RESULTS_FOLDER + 'all_data.csv', sep="\t")

# process a single day's data, this is for a given experiment, replica,
# and pinning


def process_day(filename):
    db = dbcolonysizer.DBColonySizer(show_images=None)
    day_data = db.process_files(filename)
    return day_data

# get_gene_list returns a list of positions and genes in that position. The index of
# this dataframe is the position


def get_gene_list(replicate):
    files = [
        './example_data/genes/1GS1BT.csv',
        './example_data/genes/1GS2BT.csv',
        './example_data/genes/1GS3BT.csv',
        './example_data/genes/1GS4BT.csv']
    file_order = replicate_order[replicate]
    plates_data = [pd.read_csv(i, header=1) for i in files]
    for i, plate in enumerate(plates_data):
        # delete irrelevant columns/rows
        plate.drop(plate.columns[
                   [range(6, len(plate.columns))]], axis=1, inplace=True)
        plate.drop(
            plate.index[[range(96, plate.shape[0])]], axis=0, inplace=True)
        plate.columns = [
            'ORF_original',
            'SGD_original',
            'Plate',
            'Well',
            'to_delete',
            'Control']
        plate.drop('to_delete', axis=1, inplace=True)
        plate['SGD'] = np.where(
            pd.isnull(
                plate['Control']),
            plate['SGD_original'],
            plate['Control'])
        plate['ORF'] = np.where(
            pd.isnull(
                plate['Control']),
            plate['ORF_original'],
            plate['Control'])
        plate['Column'], plate['Row'] = zip(
            *plate['Well'].apply(xl_cell_to_rowcol))
        # print plate
        if not os.path.exists(RESULTS_FOLDER + 'plate_to_gene_mappings'):
            os.makedirs(RESULTS_FOLDER + 'plate_to_gene_mappings')
        plate.to_csv(RESULTS_FOLDER + 'plate_to_gene_mappings/base_' +
                     str(os.path.basename(files[i])))
        # transform each row/column to the correct row/column on the master
        # plate
        plate['Row'] *= 2
        plate['Column'] *= 2
        print "Mapping plate " + str(i) + " to position " + str(file_order[i])
        if file_order[i] == 0:
            pass  # do nothing
        elif file_order[i] == 1:
            plate['Column'] += 1
        elif file_order[i] == 2:
            plate['Row'] += 1
        elif file_order[i] == 3:
            plate['Row'] += 1
            plate['Column'] += 1

        plate.set_index(['Row', 'Column'], inplace=True)
    master_plate = pd.concat(plates_data).sort_index()
    batch = "batch0"
    master_plate.to_csv(
        '{0}plate_to_gene_mappings/{1}_{2}_gene_mappings.csv'.format(
            RESULTS_FOLDER, batch, replicate))
    return master_plate

if __name__ == "__main__":
    process_batch('./example_data/images/batch1/')

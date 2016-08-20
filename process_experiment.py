# import the necessary packages
import numpy as np
import argparse
import pandas as pd
import os.path
from xlsxwriter.utility import xl_rowcol_to_cell, xl_cell_to_rowcol
from os.path import join
import imghdr
from glob import glob
import measure
RESULTS_FOLDER = './results_numbers/'
NUM_COLS = 24
NUM_ROWS = 16
replicate_order = {
    'A': [
        0, 1, 2, 3], 'B': [
            1, 0, 3, 2], 'C': [
                2, 3, 0, 1], 'D': [
                    3, 2, 1, 0]}  # which position each plate corresponds to on the master plate

""" Process a batch of experiments where path is a dir of experiment directories conforming to a experiment>treatment>replicate>pinning>day hierarchy where day is the leaf and is an image file."""


def measure_batch(path, fill_missing_with = None):
    # Ensure path exists for plate_measurements
    MEASUREMENTS_FOLDER = os.path.join(RESULTS_FOLDER,'plate_measurements/')
    if not os.path.exists(MEASUREMENTS_FOLDER):
        os.makedirs(MEASUREMENTS_FOLDER)
    IMAGES_FOLDER = os.path.join(RESULTS_FOLDER,'plate_images/')
    if not os.path.exists(IMAGES_FOLDER):
        os.makedirs(IMAGES_FOLDER)
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
                    root, file)) is not None]
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
    # key is replicate, values are dataframes of row/col & genes
    gene_list = {}
    gene_dict = {}  # keys are genes. used for deduping
    for replicate in replicates:
        gene_list[replicate] = get_gene_list(replicate)
        for gene in gene_list[replicate]['SGD']:
            gene_dict[gene] = 1
    genes = gene_dict.keys()

    master_table_not_defined = True
    master_table = None
    total_missing = 0

    # Process files and populate main table
    for filepath, experiment, treatment, replicate, pinning, day in data_to_process:
        day_data, missing_count = process_day(
            filepath, filename_to_save=os.path.join(IMAGES_FOLDER,"{0}_{1}_{2}_{3}_{4}.png".format(
                experiment, treatment, replicate, pinning, day)))
        day = os.path.splitext(day)[0][:-3]
        day_data.columns = ['single_column']
        day_data = day_data.join(gene_list[replicate])
        day_data.set_index('SGD', inplace=True)
        total_missing += missing_count

        # Make new index that has different levels
        day_index = pd.MultiIndex.from_tuples(
            [
                (experiment,
                 treatment,
                 replicate,
                 pinning,
                 day)],
            names=[
                'Experiments',
                'Treatments',
                'Replicates',
                'Pinnings',
                'Days'])
        day_data = day_data['single_column']
        day_data = pd.DataFrame(
            day_data.values,
            columns=day_index,
            index=day_data.index)
        groups = day_data.groupby(level=0)  # dedupe repeated indices
        day_data = groups.last()
        day_data.to_csv(
            "{0}{1}_{2}_{3}_{4}_{5}.csv".format(
                MEASUREMENTS_FOLDER,
                experiment,
                treatment,
                replicate,
                pinning,
                day),
            sep=",")
        if master_table_not_defined:

            master_table = day_data
            master_table_not_defined = False
        else:
            master_table = master_table.join(day_data, how='outer')
        print master_table.shape
    print "Total number of missing colony measurements is {0}".format(total_missing)
    if fill_missing_with is not None:
        master_table = master_table.fillna(fill_missing_with)
    return master_table


""" basic_stats_over_replicates computes and saves medians and other stats from the data set"""

def basic_stats_over_replicates(data):

    # Ensure path exists for plate_measurements
    SUMMARIES_FOLDER = os.path.join(RESULTS_FOLDER,'data_summaries/')
    if not os.path.exists(SUMMARIES_FOLDER):
        os.makedirs(SUMMARIES_FOLDER)
    data.to_csv(SUMMARIES_FOLDER + 'all_data.csv', sep=",")

    # Mean
    means = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).mean()
    means.to_csv(SUMMARIES_FOLDER + 'mean_size_over_replicates.csv', sep=",")

    # Median
    medians = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).median()
    medians.to_csv(SUMMARIES_FOLDER + 'median_size_over_replicates.csv', sep=",")

    # Standard Deviations
    stddevs = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).std()
    stddevs.to_csv(SUMMARIES_FOLDER + 'stddev_size_over_replicates.csv', sep=",")


""" compare_size computes the (median across replicates) plate normalized size of each gene from all
experiments(2,3,4,5) to the control (experiment 1) """


def compare_size(data):
    # Ensure path exists for plate_measurements
    SUMMARIES_FOLDER = os.path.join(RESULTS_FOLDER,'data_summaries/')
    if not os.path.exists(SUMMARIES_FOLDER):
        os.makedirs(SUMMARIES_FOLDER)
    data = data.divide(data.median()) # divides by the median of each plate, not of the whole df
    data = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).median()
    print "data shape {0}".format(data.shape)
    print set(list(data['EXPERIMENT2'].columns.values)) - set(list(data['EXPERIMENT1'].columns.values))
    print data['EXPERIMENT4'].columns.values
    print "data E1 {0}".format(data['EXPERIMENT1'].shape)
    print "data E2 {0}".format(data['EXPERIMENT2'].shape)
    print "data E3 {0}".format(data['EXPERIMENT3'].shape)
    print "data E4 {0}".format(data['EXPERIMENT4'].shape)
    print "data E5 {0}".format(data['EXPERIMENT5'].shape)
    data = data / pd.concat( [ data['EXPERIMENT1']]  * 5, axis=1 ).values
    #data = data[
        #['EXPERIMENT1', 'EXPERIMENT5', 'EXPERIMENT6']] / data['EXPERIMENT1']
    data.to_csv(
        SUMMARIES_FOLDER +
        'size_comparison.csv',
        sep=",")

""" process a single day's data, this is for a given experiment, replica, and pinning """

def process_day(filename, filename_to_save=None):
    db = measure.ColonyMeasurer(show_images=None, save_images="all")
    print "Processing: {0}".format(filename)
    day_data, missing_count = db.measure_colonies(
        filename, filename_to_save=filename_to_save)

    return day_data, missing_count

""" get_gene_list returns a list of positions and genes in that position. The index of this
dataframe is the position """

def get_gene_list(replicate):
    # Prepare directory
    PLATE_TO_GENE_FOLDER = os.path.join(RESULTS_FOLDER,'plate_to_gene_mappings/')
    if not os.path.exists(PLATE_TO_GENE_FOLDER):
        os.makedirs(PLATE_TO_GENE_FOLDER)
    files = [
        './example_data/genes/1GS6.csv',
        './example_data/genes/2GS1.csv',
        './example_data/genes/2GS2_alt.csv',
        './example_data/genes/2GS2.csv']
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
        plate.to_csv(os.path.join(PLATE_TO_GENE_FOLDER,
                     str(os.path.splitext(os.path.basename(files[i]))[0])+'_to_plate.csv'))
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
    master_plate.to_csv(os.path.join(PLATE_TO_GENE_FOLDER,'{0}_replicate_{1}_gene_to_master_plate_mappings.csv'.format(
            batch, replicate)))
    return master_plate

if __name__ == "__main__":
    data = measure_batch('./example_data/aug_17', fill_missing_with=5)
    data.to_pickle('./data.pkl')
    #data = pd.read_pickle('./data.pkl')
    compare_size(data)
    basic_stats_over_replicates(data)
    #day_data, missing = process_day('./example_data/Nere_imagesf/Set 1/EXPERIMENT6/T3/D/PINNING2/DAY1.JPG', filename_to_save= "w")

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


def measure_batch(path):
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
            filepath, filename_to_save="{0}_{1}_{2}_{3}_{4}".format(
                experiment, treatment, replicate, pinning, day))
        day_data.columns = ['single_column']
        day_data = day_data.join(gene_list[replicate])
        day_data.to_csv(
            "{0}{1}_{2}_{3}_{4}_{5}_original.csv".format(
                RESULTS_FOLDER,
                experiment,
                treatment,
                replicate,
                pinning,
                day),
            sep="\t")
        day_data.set_index('SGD', inplace=True)
        total_missing += missing_count

        # Make new index that has different levels
        day_index = pd.MultiIndex.from_tuples(
            [
                (experiment,
                 treatment,
                 replicate,
                 pinning,
                 os.path.splitext(day)[0])],
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
                RESULTS_FOLDER,
                experiment,
                treatment,
                replicate,
                pinning,
                day),
            sep="\t")
        if master_table_not_defined:

            master_table = day_data
            master_table_not_defined = False
        else:
            master_table = master_table.join(day_data, how='outer')
        print master_table.shape
    print "Total number of missing colony measurements is {0}".format(total_missing)
    return master_table

def print_data(data):
    # print master_table
    data.to_csv(RESULTS_FOLDER + 'all_data.csv', sep=",")
    means = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).mean()
    means.to_csv(RESULTS_FOLDER + 'all_data_means.csv', sep="\t")
    medians = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).median()
    medians.to_csv(RESULTS_FOLDER + 'all_data_medians.csv', sep="\t")
    stddevs = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).std()
    stddevs.to_csv(RESULTS_FOLDER + 'all_data_stddevs.csv', sep="\t")
    data_mean_normalized = data / data.mean()
    data_mean_normalized.to_csv(
        RESULTS_FOLDER +
        'all_data_mean_normalized.csv',
        sep="\t")
    data_median_normalized = data / data.median()
    data_median_normalized.to_csv(
        RESULTS_FOLDER +
        'all_data_median_normalized.csv',
        sep="\t")
    median_normalized_means = data_median_normalized.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).mean()
    median_normalized_means.to_csv(
        RESULTS_FOLDER +
        'median_normalized_means.csv',
        sep="\t")
    median_normalized_medians = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).median()
    median_normalized_medians.to_csv(
        RESULTS_FOLDER +
        'median_normalized_medians.csv',
        sep="\t")

    #control_normalized_comparisons = median_normalized_means[
        #['EXPERIMENT1', 'EXPERIMENT5', 'EXPERIMENT6']] / median_normalized_means['EXPERIMENT1']
    #control_normalized_comparisons.to_csv(
        #RESULTS_FOLDER +
        #'control_normalized_comparisons.csv',
        #sep="\t")
    print means
    print medians
    # print means_normalized
    # print medians_normalized
    #means_normalized.to_csv(RESULTS_FOLDER + 'all_data_means.csv', sep="\t")
    #medians_normalized.to_csv(RESULTS_FOLDER + 'all_data_medians.csv', sep="\t")

""" compare_size computes the (median across replicates) plate normalized size of each gene from all
experiments(2,3,4,5) to the control (experiment 1) """

def compare_size(data):
    data = data.divide(data.median())
    data = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).median()
    #df2 = data.copy()
    #for c in data.columns.levels[0]:
            #df2[c] = df[c] / df['three']
    data = data / pd.concat( [ data['EXPERIMENT1']]  * 5, axis=1 ).values
    #data = data[
        #['EXPERIMENT1', 'EXPERIMENT5', 'EXPERIMENT6']] / data['EXPERIMENT1']
    data.to_csv(
        RESULTS_FOLDER +
        'size_comparison.csv',
        sep="\t")

# process a single day's data, this is for a given experiment, replica,
# and pinning


def process_day(filename, filename_to_save=None):
    db = measure.ColonyMeasurer()#show_images="all", save_images="all")
    print "Processing: {0}".format(filename)
    day_data, missing_count = db.measure_colonies(
        filename, filename_to_save=filename_to_save)

    return day_data, missing_count

# get_gene_list returns a list of positions and genes in that position. The index of
# this dataframe is the position


def get_gene_list(replicate):
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
    data = measure_batch('./example_data/second_data')
    data.to_pickle('./data.pkl')
    #data = read_pickle('./data.pkl')
    compare_size(data)
    #day_data, missing = process_day('./example_data/Nere_imagesf/Set 1/EXPERIMENT6/T3/D/PINNING2/DAY1.JPG', filename_to_save= "w")
    # print day_data

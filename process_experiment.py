# import the necessary packages
import numpy as np
import re
import argparse
import pandas as pd
import os.path
from xlsxwriter.utility import xl_rowcol_to_cell, xl_cell_to_rowcol
from os.path import join
import imghdr
import measure

class SetMeasurer(object):
    def __init__(self, root_path, gene_folder):
        self.gene_folder = gene_folder
        self.root_path = path 
        self.file
        
        self.pinnings = []
        self.replicates = []
        self.experiments = []
        self.treatments = []
        self.days = []
        self.data_to_process = []

        self.NUM_COLS = 24
        self.NUM_ROWS = 16
        self.replicate_order = {
            'A': [
                0, 1, 2, 3], 'B': [
                    1, 0, 3, 2], 'C': [
                        2, 3, 0, 1], 'D': [
                            3, 2, 1, 0]}  # which position each plate corresponds to on the master plate

        self.RESULTS_FOLDER = './results_numbers/'
        if not os.path.exists(self.RESULTS_FOLDER):
            os.makedirs(self.RESULTS_FOLDER)

        self.MEASUREMENTS_FOLDER = os.path.join(self.RESULTS_FOLDER, 'plate_measurements/')
        if not os.path.exists(self.MEASUREMENTS_FOLDER):
            os.makedirs(self.MEASUREMENTS_FOLDER)
        
        self.IMAGES_FOLDER = os.path.join(self.RESULTS_FOLDER, 'plate_images/')
        if not os.path.exists(self.IMAGES_FOLDER):
            os.makedirs(self.IMAGES_FOLDER)

    """ read_set operates on a directory for a set and populates all the internal variables based on what it reads """
    def read_set():
        pinnings = {}
        replicates = {}
        experiments = {}
        treatments = {}
        days = {}
        data_to_process = []
        for root, dirs, files in os.walk(self.root_path):
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
            assert re.match(
                r"^PINNING \d$", pinning), "Invalid pinning at {0}, interpreted pinning to be \"{1}\"".format(
                root, pinning)
            assert re.match(
                r"^[A-Z]$", replicate), "Invalid replicate at {0}, interpreted replicate to be \"{1}\"".format(
                root, replicate)
            assert re.match(
                r"^T\d$", treatment), "Invalid treatment at {0}, interpreted treatment to be \"{1}\"".format(
                root, treatment)
            assert re.match(
                r"^EXPERIMENT \d$", experiment), "Invalid experiment at {0}, interpreted experiment to be \"{1}\"".format(
                root, experiment)
            pinnings[pinning] = 1
            replicates[replicate] = 1
            treatments[treatment] = 1
            experiments[experiment] = 1
            for day in files:
                filepath = os.path.join(root, day)
                assert re.match(r"^DAY\d_", os.path.splitext(
                    day)[0]), "Invalid day at {0}, interpreted day to be \"{1}\"".format(root, day)
                cleaned_day = os.path.splitext(day)[0][:4]
                days[cleaned_day] = 1
                data_to_process.append(
                    (filepath, experiment, treatment, replicate, pinning, cleaned_day))
        assert len(
            data_to_process) > 0, "No valid files found to process. Check the directory."
        self.pinnings = pinnings.keys()
        self.replicates = replicates.keys()
        self.experiments = experiments.keys()
        self.treatments = treatments.keys()
        self.days = days.keys()
        self.data_to_process = data_to_process

    """ Process a batch of experiments where path is a dir of experiment directories conforming to a 
    experiment>treatment>replicate>pinning>day hierarchy where day is the leaf and is an image file."""
    def measure_batch(path, gene_folder, fill_missing_with=None):
        print "Processing {0} files".format(len(data_to_process))
        print "Experiments: " + str(self.experiments)
        print "Treatments: " + str(self.treatments)
        print "Replicates: " + str(self.replicates)
        print "Pinnings: " + str(self.pinnings)
        print "Days: " + str(self.days)
        print "Data to process: " + str(self.data_to_process)
        c_iterables = [self.experiments, self.treatments, self.replicates, self.pinnings, self.days]
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
        for replicate in self.replicates:
            gene_list[replicate] = get_gene_list(replicate, gene_folder)
            for gene in gene_list[replicate]['SGD']:
                gene_dict[gene] = 1
        genes = gene_dict.keys()

        master_table_not_defined = True
        master_table = None
        total_missing = 0

        # Process files and populate main table
        for filepath, experiment, treatment, replicate, pinning, day in data_to_process:
            day_data, missing_count = process_day(
                filepath, 
                filename_to_save=os.path.join(
                    IMAGES_FOLDER, "{0}_{1}_{2}_{3}_{4}.png".format(
                        experiment, treatment, replicate, pinning, day)),
                gene_names=gene_list[replicate])
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

        # fill in missing columns so we can easily compare later
        for experiment in experiments:
            # loop over all t,r,p,d for all experiments
            for _, _, treatment, replicate, pinning, day in data_to_process:
                if (experiment,
                        treatment,
                        replicate,
                        pinning,
                        day) not in master_table.columns:
                    print("There were no measurements for [{0},{1},{2},{3},{4}]".format(
                        experiment,
                        treatment,
                        replicate,
                        pinning,
                        day))

                    master_table[
                        experiment,
                        treatment,
                        replicate,
                        pinning,
                        day] = np.nan

        return master_table


""" basic_stats_over_replicates computes and saves medians and other stats from the data set"""


def basic_stats_over_replicates(data):

    # Ensure path exists for plate_measurements
    SUMMARIES_FOLDER = os.path.join(RESULTS_FOLDER, 'data_summaries/')
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
    medians.to_csv(
        SUMMARIES_FOLDER +
        'median_size_over_replicates.csv',
        sep=",")

    # Standard Deviations
    stddevs = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).std()
    stddevs.to_csv(
        SUMMARIES_FOLDER +
        'stddev_size_over_replicates.csv',
        sep=",")


""" compare_size computes the (median across replicates) plate normalized size of each gene from all
experiments(2,3,4,5) to the control (experiment 1) """


def compare_size(data):
    # Ensure path exists for plate_measurements
    SUMMARIES_FOLDER = os.path.join(RESULTS_FOLDER, 'data_summaries/')
    if not os.path.exists(SUMMARIES_FOLDER):
        os.makedirs(SUMMARIES_FOLDER)
    # divides by the median of each plate, not of the whole df
    data = data.divide(data.median())
    data = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).median()
    data = data / pd.concat([data['EXPERIMENT 1']] *
                            data.columns.get_level_values(0).unique().size, axis=1).values
    # data = data[
    data.to_csv(
        SUMMARIES_FOLDER +
        'size_comparison.csv',
        sep=",")

""" compute_gene_summary computes the (median across replicates / days / pinnings) plate normalized size of each gene from all
experiments(2,3,4,5) and divides by the control (experiment 1) """


def compute_gene_summary(data):
    # Ensure path exists for plate_measurements
    SUMMARIES_FOLDER = os.path.join(RESULTS_FOLDER, 'data_summaries/')
    if not os.path.exists(SUMMARIES_FOLDER):
        os.makedirs(SUMMARIES_FOLDER)
    # divides by the median of each plate, not of the whole df
    data = data.divide(data.median())
    data = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments',
            'Pinnings',
            'Days']).median()
    # Compare to Exp1
    data = data / pd.concat([data['EXPERIMENT 1']] *
                            data.columns.get_level_values(0).unique().size, axis=1).values
    data = data.groupby(
        axis=1,
        level=[
            'Experiments',
            'Treatments']).median()
    data = data.stack()
    data.to_pickle("/tmp/pckl")
    # data = data[
    data.to_csv(
        SUMMARIES_FOLDER +
        'gene_summary.csv',
        sep=",")

""" process a single day's data, this is for a given experiment, replica, and pinning """


def process_day(filename, filename_to_save=None, gene_names=None):
    db = measure.PlateMeasurer(show_images="all", save_images="all",gene_names=gene_names)
    print "Processing: {0}".format(filename)
    day_data, missing_count = db.process_plate_image(
        filename, filename_to_save=filename_to_save)

    return day_data, missing_count

""" get_gene_list returns a list of positions and genes in that position. The index of this
dataframe is the position """


def get_gene_list(replicate, genes_folder):
    # Prepare directory
    PLATE_TO_GENE_FOLDER = os.path.join(
        RESULTS_FOLDER,
        'plate_to_gene_mappings/')
    if not os.path.exists(PLATE_TO_GENE_FOLDER):
        os.makedirs(PLATE_TO_GENE_FOLDER)
    files = []
    assert os.path.isdir(
        genes_folder), "Specified folder {0} is not a folder".format(genes_folder)
    for root, dirs, file_list in os.walk(genes_folder):
        # keep only images
        for f in file_list:
            if os.path.splitext(f)[1] == '.csv':
                files.append(os.path.join(root, f))
    files = sorted(files)
    assert(len(files) == 4)
    file_order = replicate_order[replicate]
    plates_data = [pd.read_csv(i, header=0, engine='python') for i in files]
    for i, plate in enumerate(plates_data):
        # delete irrelevant columns/rows
        plate.drop(plate.columns[
                   [range(6, len(plate.columns))]], axis=1, inplace=True)
        plate.drop(
            plate.index[[range(96, plate.shape[0])]], axis=0, inplace=True)
        plate.columns = [
            'ORF',
            'SGD',
            'Plate',
            'Well',
            'to_delete',
            'Control']
        print (plate)
        plate.drop('to_delete', axis=1, inplace=True)
        plate['Column'], plate['Row'] = zip(
            *plate['Well'].apply(xl_cell_to_rowcol))
        plate.to_csv(
            os.path.join(
                PLATE_TO_GENE_FOLDER, str(
                    os.path.splitext(
                        os.path.basename(
                            files[i]))[0]) + '_to_plate.csv'))
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
        os.path.join(
            PLATE_TO_GENE_FOLDER,
            '{0}_replicate_{1}_gene_to_master_plate_mappings.csv'.format(
                batch,
                replicate)))
    return master_plate


def process(data_directory, gene_directory):  # , kernel_image):
    m = SetMeasurer(data_directory, gene_directory)
    m.read_set()
    data = m.measure_batch(fill_missing_with=5)
    data.to_pickle('./data.pkl')
    #data = pd.read_pickle('./data.pkl')
    compare_size(data)
    basic_stats_over_replicates(data)

if __name__ == "__main__":
    data = measure_batch(
        './example_data/aug_22',
        './example_data/genes',
        fill_missing_with=5)
    data.to_pickle('./data.pkl')
    #data = pd.read_pickle('./data.pkl')
    compare_size(data)
    compute_gene_summary(data)

    basic_stats_over_replicates(data)
    print ("Complete! You can close this window now.")

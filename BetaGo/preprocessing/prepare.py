from sgf_crawler import run_sgf_crawler
from hdf5_converter import run_hdf5_converter
from hdf5_merger import run_hdf5_merger
import os
import shutil


def run_prepare():

    # stage1: download sgfs

    # stage2: convert new sgf files to new_version hdf5 file
    args = ['--features', all,
            '--outfile', '../../data/hdf5s/new_version.hdf5',
            '--directory', '../../data/sgfs', '--recurse']
    run_hdf5_converter(args)

    # stage3: merge new_version and old_version to active_version
    args = ['--features', all,
            '--outfile', '../../data/active_version.hdf5',
            '--directory', '../../data/hdf5s', '--recurse']
    run_hdf5_merger(args)
    os.remove('../../data/hdf5s/new_version.hdf5')
    os.remove('../../data/hdf5s/old_version.hdf5')
    shutil.copyfile('../../data/active_version.hdf5', '../../data/hdf5s/old_version.hdf5')

if __name__ == '__main__':
    run_prepare()
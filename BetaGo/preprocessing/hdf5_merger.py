import h5py as h5
import numpy as np
import os
import warnings


class HDF5_merger:

    def __init__(self):
        self.n_features = 47

    """ Convert all src hdf5 files into an hdf5 file
        Arguments:
        - src_hdf5_files : an iterable of relative or absolute paths to SGF files
        - des_hdf5_file : the name of the HDF5 where features will be saved
        - board_size : side length of board of games that are loaded
        - ignore_errors : if True, issues a Warning when there is an unknown exception rather than halting.
        Note that sgf.ParseException and go.IllegalMove exceptions are always skipped

        The resulting file has the following properties:
            states  : dataset with shape (n_data, n_features, board width, board height)
            actions : dataset with shape (n_data, 2) (actions are stored as x,y tuples of where the move was played)
            file_offsets : group mapping from filenames to tuples of (index, length)

        For example, to find what positions in the dataset come from 'test.sgf':
            index, length = file_offsets['test.sgf']
            test_states = states[index:index+length]
            test_actions = actions[index:index+length]
    """
    def policyNet_merge(self, src_hdf5_files, des_hdf5_file, board_size=19, ignore_errors=True, verbose=False):

        # TODO - also save feature list
        # make a hidden temporary file in case of a crash.
        # on success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(des_hdf5_file), ".tmp." + os.path.basename(des_hdf5_file))
        h5file = h5.File(tmp_file, 'w')

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            # h5py.require_dataset() : Open a dataset if exists, otherwise, create it.
            # h5py.create_dataset() : Create a dataset, if exists, overwrite it.
            # h5py.require_group() :
            # h5py.create_group()
            states = h5file.require_dataset('states',
                dtype=np.uint8,
                shape=(1, self.n_features, board_size, board_size),
                maxshape=(None, self.n_features, board_size, board_size),  # 'None' dimension allows it to grow arbitrarily
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, self.n_features, board_size, board_size),  # approximately 1MB chunks
                compression="lzf")
            actions = h5file.require_dataset('actions',
                dtype=np.uint8,
                shape=(1, 2),
                maxshape=(None, 2),
                exact=False,
                chunks=(1024, 2),
                compression="lzf")
            # 'file_offsets' is an HDF5 group so that 'file_name in file_offsets' is fast
            file_offsets = h5file.require_group('file_offsets')

            if verbose:
                print("created HDF5 dataset in {}".format(tmp_file))

            next_file_index = 0
            for file_path in src_hdf5_files:
                if verbose:
                    print(file_path)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                file_start_index = next_file_index
                try:
                    dataset = h5.File(file_path)

                    temp_next_file_index1 = next_file_index
                    for state in dataset["states"]:
                        if temp_next_file_index1 >= len(states):
                            states.resize((temp_next_file_index1 + 1, self.n_features, board_size, board_size))
                        states[temp_next_file_index1] = state
                        temp_next_file_index1 += 1

                    temp_next_file_index2 = next_file_index
                    for action in dataset["actions"]:
                        if temp_next_file_index2 >= len(actions):
                            actions.resize((temp_next_file_index2 + 1, 2))
                        actions[temp_next_file_index2] = action
                        temp_next_file_index2 += 1

                    if temp_next_file_index1 == temp_next_file_index2:
                        n_pairs = next_file_index - temp_next_file_index1
                        next_file_index = temp_next_file_index1
                    else:
                        raise Exception

                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_path, e), stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        # '/' has special meaning in HDF5 key names, so they are replaced with
                        # ':' here
                        file_name_key = file_path.replace('/', ':')
                        file_offsets[file_name_key] = [file_start_index, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("sgfs_to_hdf5 failed due to ", e.message)
            os.remove(tmp_file)
            raise e

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, des_hdf5_file))

        # processing complete; rename tmp_file to hdf5_file
        h5file.close()
        os.rename(tmp_file, des_hdf5_file)

    """ Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file

            Arguments:
            - sgf_files : an iterable of relative or absolute paths to SGF files
            - hdf5_file : the name of the HDF5 where features will be saved
            - board_size : side length of board of games that are loaded
            - ignore_errors : if True, issues a Warning when there is an unknown exception rather than halting. Note
                that sgf.ParseException and go.IllegalMove exceptions are always skipped

            The resulting file has the following properties:
                states  : dataset with shape (n_data, n_features, board width, board height)
                actions : dataset with shape (n_data, 2) (actions are stored as x,y tuples of where the move was played)
                file_offsets : group mapping from filenames to tuples of (index, length)

            For example, to find what positions in the dataset come from 'test.sgf':
                index, length = file_offsets['test.sgf']
                test_states = states[index:index+length]
                test_actions = actions[index:index+length]
    """
    # TODO - also save feature list
    def valueNet_merge(self, src_hdf5_files, des_hdf5_file, board_size=19, ignore_errors=True, verbose=False):

        # make a hidden temporary file in case of a crash.
        # on success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(des_hdf5_file), ".tmp." + os.path.basename(des_hdf5_file))
        h5file = h5.File(tmp_file, 'w')

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            # h5py.require_dataset() : Open a dataset if exists, otherwise, create it.
            # h5py.create_dataset() : Create a dataset, if exists, overwrite it.
            # h5py.require_group() :
            # h5py.create_group()
            states = h5file.require_dataset('states',
                dtype=np.uint8,
                shape=(1, self.n_features, board_size, board_size),
                maxshape=(None, self.n_features, board_size, board_size),  # 'None' dimension allows it to grow arbitrarily
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, self.n_features, board_size, board_size),  # approximately 1MB chunks
                compression="lzf")

            winners = h5file.require_dataset(
                'winners',
                dtype=np.int8,
                shape=(1, 1),
                maxshape=(None, 1),
                exact=False,
                chunks=(1024, 1),
                compression="lzf")
            # 'file_offsets' is an HDF5 group so that 'file_name in file_offsets' is fast
            file_offsets = h5file.require_group('file_offsets')

            if verbose:
                print("created HDF5 dataset in {}".format(tmp_file))

            next_file_index = 0
            for file_path in src_hdf5_files:
                if verbose:
                    print(file_path)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                file_start_index = next_file_index
                try:
                    dataset = h5.File(file_path)

                    temp_next_file_index1 = next_file_index
                    for state in dataset["states"]:
                        if temp_next_file_index1 >= len(states):
                            states.resize((temp_next_file_index1 + 1, self.n_features, board_size, board_size))
                        states[temp_next_file_index1] = state
                        temp_next_file_index1 += 1

                    temp_next_file_index2 = next_file_index
                    for winner in dataset["winners"]:
                        if temp_next_file_index2 >= len(winners):
                            winners.resize((temp_next_file_index2 + 1, 1))
                        winners[temp_next_file_index2] = winner
                        temp_next_file_index2 += 1

                    if temp_next_file_index1 == temp_next_file_index2:
                        n_pairs = temp_next_file_index1 - next_file_index
                        next_file_index = temp_next_file_index1
                    else:
                        raise Exception

                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_path, e), stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        # '/' has special meaning in HDF5 key names, so they are replaced with
                        # ':' here
                        file_name_key = file_path.replace('/', ':')
                        file_offsets[file_name_key] = [file_start_index, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("merge hdf5 files failed due to ", e.message)
            os.remove(tmp_file)
            raise e

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, des_hdf5_file))

        # processing complete; rename tmp_file to hdf5_file
        h5file.close()
        os.rename(tmp_file, des_hdf5_file)


def _is_hdf5(filename):
    return filename.strip()[-3:] == ".h5"


def _list_hdf5(path):
    files = os.listdir(path)
    return (os.path.join(path, f) for f in files if _is_hdf5(f))


def _walk_all_hdf5(path):
    for (dirpath, dirname, files) in os.walk(path):
        return _list_hdf5(dirpath)


def run_hdf5_merger(cmd_line_args=None):
    """Run merge. command-line args may be passed in as a list
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge multi hdf5 files into one for training the neural network model.',
        epilog="Available features are: board, ones, turns_since, liberties,\
        capture_size, self_atari_size, liberties_after, sensibleness, and zeros.\
        Ladder features are not currently implemented")
    parser.add_argument("--outfile", "-o",
                        help="Destination to write hdf5 file",
                        default="d:\\dodi\\BetaGo\\data\\merged.hdf5")
    parser.add_argument("--directory", "-d",
                        help="Directory containing hdf5 files to process.",
                        default="d:\\dodi\\data\\hdf5s\\")
    parser.add_argument("--recurse", "-r",
                        help="Set to recurse through directories searching for hdf5 files",
                        default=False, action="store_true")  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode",
                        default=False, action="store_true")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # get an iterator of SGF files according to command line args
    if args.directory:
        if args.recurse:
            src_hdf5_files = _walk_all_hdf5(args.directory)
        else:
            src_hdf5_files = _list_hdf5(args.directory)
    des_hdf5_file = args.outfile
    merger = HDF5_merger()
    merger.valueNet_merge(src_hdf5_files, des_hdf5_file,
                          board_size=19, ignore_errors=False, verbose=True);


if __name__ == '__main__':
    run_hdf5_merger()

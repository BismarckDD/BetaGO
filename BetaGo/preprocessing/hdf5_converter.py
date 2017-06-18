import numpy as np
import h5py as h5
import os
import warnings
import sys
sys.path.append("D:\dodi\BetaGo")
from BetaGo.preprocessing.preprocessing import Preprocess
from BetaGo.util import sgf_iter_states
import BetaGo.go as go
import sgf
import multiprocessing
from multiprocessing.pool import Pool


class SizeMismatchError(Exception):
    pass

worker_pool = None


class HDF5_converter:

    def __init__(self, features):
        self.feature_processor = Preprocess(features)
        self.n_features = self.feature_processor.output_dim

    """ Read the given SGF file into an iterable of (input,output) pairs for neural network training
        Each input is a GameState converted into one-hot neural net features
        Each output is an action as an (x,y) pair (passes are skipped)
        If this game's size does not match board_size, a SizeMismatchError is raised """
    def convert_sgf_to_gamestate_iter(self, file_name, board_size):

        with open(file_name, 'r') as file_object:
            state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)

        for (state, move, player) in state_action_iterator:
            if state.size != board_size:
                raise SizeMismatchError()
            if move != go.PASS_MOVE:
                nn_input = self.feature_processor.state_to_tensor(state)
                yield (nn_input, move)

    """ Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file
        Arguments:
            - sgf_files : an iterable of relative or absolute paths to SGF files
            - hdf5_file : the name of the HDF5 where features will be saved
            - board_size : side length of board of games that are loaded

            - ignore_errors : if True, issues a Warning when there is an unknown
                exception rather than halting. Note that sgf.ParseException and
                go.IllegalMove exceptions are always skipped

            The resulting file has the following properties:
                states  : dataset with shape (n_data, n_features, board width, board height)
                actions : dataset with shape (n_data, 2) (actions are stored as x,y tuples of
                          where the move was played)
                file_offsets : group mapping from filenames to tuples of (index, length)

            For example, to find what positions in the dataset come from 'test.sgf':
                index, length = file_offsets['test.sgf']
                test_states = states[index:index+length]
                test_actions = actions[index:index+length]
     """
    # TODO - also save feature list
    def sgfs_to_hdf5(self, sgf_files, hdf5_file, board_size=19, ignore_errors=True, verbose=False):

        # make a hidden temporary file in case of a crash.
        # on success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5file = h5.File(tmp_file, 'w')

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            # h5py.require_dataset() : Open a dataset if exists, otherwise, create it.
            # h5py.create_dataset() : Create a dataset, if exists, overwrite it.
            # h5py.require_group() :
            # h5py.create_group() :
            states = h5file.require_dataset(
                'states',
                dtype=np.uint8,
                shape=(1, self.n_features, board_size, board_size),
                maxshape=(None, self.n_features, board_size, board_size),  # 'None' == arbitrary size
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, self.n_features, board_size, board_size),  # approximately 1MB chunks
                compression="lzf")

            actions = h5file.require_dataset(
                'actions',
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
            for file_name in sgf_files:
                if verbose:
                    print(file_name)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                start_file_index = next_file_index
                try:
                    for state, move in self.convert_sgf_to_gamestate_iter(file_name, board_size):
                        if next_file_index >= len(states):
                            states.resize((next_file_index + 1, self.n_features, board_size, board_size))
                            actions.resize((next_file_index + 1, 2))
                        states[next_file_index] = state
                        actions[next_file_index] = move
                        n_pairs += 1
                        next_file_index += 1
                except go.IllegalMove:
                    warnings.warn("Illegal Move encountered in %s\n"
                                  "\tdropping the remainder of the game" % file_name)
                except sgf.ParseException:
                    warnings.warn("Could not parse %s\n\tdropping game" % file_name)
                except SizeMismatchError:
                    warnings.warn("Skipping %s; wrong board size" % file_name)
                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e), stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        # '/' has special meaning in HDF5 key names, so they
                        # are replaced with ':' here
                        file_name_key = file_name.replace('/', ':')
                        file_offsets[file_name_key] = [start_file_index, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("sgfs_to_hdf5 failed")
            print e
            os.remove(tmp_file)
            raise e

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        # processing complete; rename tmp_file to hdf5_file
        h5file.close()
        os.rename(tmp_file, hdf5_file)


def _is_sgf(filename):
    return filename.strip()[-4:] == ".sgf"


def _walk_all_sgfs(path):
    result = []
    for (dirpath, subdirs, files) in os.walk(path):
        result.extend(_list_sgfs(dirpath))
    return result


def _list_sgfs(path):
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files if _is_sgf(f)]


def helper_function(feature_list, sgf_files, outfile_name, size, verbose):
    print outfile_name
    converter = HDF5_converter(feature_list)
    converter.sgfs_to_hdf5(sgf_files, outfile_name, board_size=size, verbose=verbose)
    return


def run_hdf5_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Prepare hdf5 files for training the neural network model.',
        epilog="Available features are: board, ones, turns_since, liberties,\
        capture_size, self_atari_size, liberties_after, sensibleness, and zeros.\
        Ladder features are not currently implemented")
    parser.add_argument("--features", "-f",
                        help="Comma-separated list of features to compute and store or 'all'",
                        default='all')
    parser.add_argument("--outfile", "-o",
                        help="Destination to write data (hdf5 file)",
                        default="d:\\dodi\\BetaGo\\data\\hdf5s\\new_version.hdf5")
    parser.add_argument("--directory", "-d",
                        help="Directory containing SGF files to process.",
                        default="d:\\dodi\\BetaGo\\data\\sgfs\\")
    parser.add_argument("--size", "-s",
                        help="Size of the game board. SGFs not matching this are discarded with a warning",
                        type=int, default=19)
    parser.add_argument("--recurse", "-r",
                        help="Set to recurse through directories searching for SGF files",
                        default=False, action="store_true")
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode",
                        default=False, action="store_true")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.features.lower() == 'all':
        feature_list = [
            "board",
            "ones",
            "turns_since",
            "liberties",
            "capture_size",
            "self_atari_size",
            "liberties_after",
            # "ladder_capture",
            # "ladder_escape",
            "sensibleness",
            "zeros"]
    else:
        feature_list = args.features.split(",")

    if args.verbose:
        print("using features", feature_list)

    # converter = HDF5_converter(feature_list)

    # get an iterator of SGF files according to command line args
    if args.directory:
        # print "Enter"
        if args.recurse:
            # print "Enter"
            sgf_files = _walk_all_sgfs(args.directory)
        else:
            sgf_files = _list_sgfs(args.directory)

    # print args.recurse, args.directory, type(sgf_files)

    # def test_walk_all_sgfs(path):
    #     for (dirpath, subdirs, files) in os.walk(path):
    #         print dirpath, subdirs

    # test_walk_all_sgfs(args.directory)
    # for sgf_file in _walk_all_sgfs(args.directory):
    #     print sgf_file

    """
    # Check if it gets all sgf files.
    for (dirpath, subdirs, files) in os.walk(args.directory):
        files = os.listdir(dirpath)
        for file in files:
            if _is_sgf(file):
                print os.path.join(dirpath, file)
    """

    workers_num = multiprocessing.cpu_count() if not args.verbose else 1  # set to 1 when debugging
    global worker_pool
    if worker_pool is None:
        worker_pool = Pool(processes=workers_num)
    results = []
    sgf_files_num = len(sgf_files)
    files_per_worker = (sgf_files_num // workers_num) + 1
    for i in xrange(workers_num):
        outfile_name = args.outfile.split(".")[0] + str(i*files_per_worker) + "_" + str((i+1)*files_per_worker) + ".hdf5"
        print outfile_name
        results.append(worker_pool.apply_async(helper_function, (feature_list,
            sgf_files[i*files_per_worker:(i+1)*files_per_worker], outfile_name, args.size, args.verbose)))
    worker_pool.close()
    worker_pool.join()
    # while results:
    #     for result in results:
    #         if result.ready():
    #             results.remove(result)
    #     time.sleep(10)

    print("HDF5s have been generated!")


if __name__ == '__main__':
    run_hdf5_converter()

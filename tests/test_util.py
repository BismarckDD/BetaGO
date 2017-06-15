import unittest
import os
import sgf
import sys
sys.path.append("..")
from BetaGo.util import sgf_iter_states


def is_sgf(filename):
    return filename.strip()[-4:] == ".sgf"


def walk_all_sgfs(path):
    for (dirpath, subdirs, files) in os.walk(path):
        return list_sgfs(dirpath)


def list_sgfs(path):
    files = os.listdir(path)
    for file in files:
        if is_sgf(file):
            yield os.path.join(path, file)


class TestUtil(unittest.TestCase):

    def test_util(self):
        path = "D:\\dodi\\BetaGo\\tests\\test_data\\sgf"
        """
        for (dirpath, subdirs, files) in os.walk(path):
            print dirpath, subdirs, files
        """
        # files = list_sgfs(path)
        files = walk_all_sgfs(path)
        for file_name in files:
            with open(file_name, 'r') as file_object:
                collection = sgf.parse(file_object.read())
                print collection[0].rest
                prop = collection[0].root.properties
                print str(prop) #properties is a dict
                state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)
                print state_action_iterator


# Maybe use register mechanism to run all tests.
if __name__ == "__main__":
    unittest.main()
import os
import sys
sys.path.append("..")
from BetaGo.training.supervised_policy_trainer import run_training
import unittest


class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        model = 'test_data/minimodel.json'
        data = 'test_data/hdf5/alphago-vs-lee-sedol-features.hdf5'
        output = 'test_data/.tmp.training/'
        args = [model, data, output, '--epochs', '1']
        run_training(args)

        os.remove(os.path.join(output, 'metadata.json'))
        os.remove(os.path.join(output, 'shuffle.npz'))
        os.remove(os.path.join(output, 'weights.00000.hdf5'))
        os.rmdir(output)

if __name__ == '__main__':
    unittest.main()

import unittest
import os
import subprocess


class TestMain(unittest.TestCase):
    """Testing the entire pipeline on a small example dataset """
    def __init__(self, *args, **kwargs):
        super(TestMain, self).__init__(*args, **kwargs)

    def test_main_full_cycle(self):
        """Test with both pretraining DAMSM and attnGAN training"""
        output = subprocess.Popen(['python', './main.py', '-c', '../test/config/full_train.ini', '--train'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        stdout, stderr = output.communicate()
        # print(stderr)
        if stderr:
            self.fail("Full train cycle failed with error:\n\n{}".format(stderr.decode()))

    def test_main_damsm_pretrain(self):
        """Test with only pretraining DAMSM"""
        output = subprocess.Popen(['python', './main.py', '-c', '../test/config/damsm_pretrain.ini', '--train'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        stdout, stderr = output.communicate()
        # print(stderr)
        if stderr:
            self.fail("DAMSM pretrain cycle failed with error:\n\n{}".format(stderr.decode()))

    def test_main_attngan_train(self):
        """Test with attnGAN training and DAMSM objects loading"""
        output = subprocess.Popen(['python', './main.py', '-c', '../test/config/attngan_train.ini', '--train'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        stdout, stderr = output.communicate()
        if stderr:
            self.fail("AttnGAN train cycle failed with error:\n\n{}".format(stderr.decode()))

if __name__ == "__main__":
    unittest.main()
    
import unittest

from ..utils import clean_str, clean_str_sst


class CleanStrTest(unittest.TestCase):
    def test_clean_str(self):
        dummy_str = '[Hello, my name is9 () don\'t! 고구마  '
        print(clean_str(dummy_str))

    def test_clean_str_sst(self):
        dummy_str = '[Hello, my name is9 () don\'t! 고구마  '
        print(clean_str_sst(dummy_str))


from Project03CornerStone import *
import unittest


class GedTest(unittest.TestCase):

    def test_Marriage_before_divorce(self, family):
        for i in marriage_before_divorce(family):
            if i[4] != "NA":
                self.assertLess(datetime.strptime(i[3], "%Y %b %d"), datetime.strptime(i[4], "%Y %b %d"))

    def test_Divorce_before_death(self, family, individual):
        for i in divorce_before_death(family, individual):
            self.assertEqual(i[4], "NA")


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
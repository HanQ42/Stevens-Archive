import unittest
from The_Real_Project03 import *

class Test_case_us33_us34(unittest.TestCase):
    def setUp(self):
        file_path = '555Project(updates-often).ged'
        '''Dictionaries of Individuals and Families'''
        individuals = {}
        families = {}
        # 1. Read the file
        with open(file_path, 'r') as file:
            unfiltered_file = file.readlines()
            # 2. Filter the file
            filter_file(unfiltered_file, individuals, families)

        individual_dict = {}
        family_dict = {}
        '''the next 2 lines fill in the previous 2 lines individual_dict and family_dict'''
        raw_individuals_to_structured_dict(individuals, individual_dict)
        raw_families_to_structured_dict(families, family_dict)


        '''Xiangyu Sprint 1: US15, US24'''
        self.US33_report = list_orphans(family_dict, individual_dict)
        self.US34_report = list_large_age_differences(family_dict, individual_dict)

    def test_US33(self):
        expect = {'@I58@': ['2017-2-2', '2016-3-3', 14],
                  '@I57@': ['2017-2-2', '2016-3-3', 16],
                  '@I56@': ['2017-2-2', '2016-3-3', 17]}

        actual = self.US33_report
        for key in expect.keys():
            self.assertEqual(actual[key], expect[key])


    def test_US34(self):
        expect = {'@F8@': [60, 30],
                  '@F19@': [61, 26]}
        actual = self.US34_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())

if __name__ == '__main__':
    unittest.main()

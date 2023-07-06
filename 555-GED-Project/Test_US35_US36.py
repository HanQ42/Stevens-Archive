import unittest
from The_Real_Project03 import *
import datetime

class Test_case_us35_us36(unittest.TestCase):
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


        '''Shengda Sprint 3: US15, US24'''
        self.US35_report = List_recent_births(individual_dict)
        self.US36_report = List_recent_deaths(individual_dict)

    def test_US35(self):
        expect = {'@I67@': [datetime.datetime(2020, 4, 3, 0, 0), datetime.datetime(2020, 4, 6, 10, 46, 12, 302546)],
                  '@I68@': [datetime.datetime(2020, 4, 3, 0, 0), datetime.datetime(2020, 4, 6, 10, 46, 12, 302546)]}

        actual = self.US35_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())


    def test_US36(self):
        expect = {'@I64@': [datetime.datetime(2020, 4, 1, 0, 0), datetime.datetime(2020, 4, 6, 10, 46, 12, 303477)],
                  '@I68@': [datetime.datetime(2020, 4, 4, 0, 0), datetime.datetime(2020, 4, 6, 10, 46, 12, 303477)],
                  '@I69@': [datetime.datetime(2020, 3, 22, 0, 0), datetime.datetime(2020, 4, 6, 10, 46, 12, 303477)],
                  '@I70@': [datetime.datetime(2020, 3, 22, 0, 0), datetime.datetime(2020, 4, 6, 10, 46, 12, 303477)],
                  '@I71@': [datetime.datetime(2020, 3, 22, 0, 0), datetime.datetime(2020, 4, 6, 10, 46, 12, 303477)]}

        actual = self.US36_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())

if __name__ == '__main__':
    unittest.main()

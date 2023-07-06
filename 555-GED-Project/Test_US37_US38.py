import unittest
from The_Real_Project03 import *
import datetime

class Test_case_us37_us38(unittest.TestCase):
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


        '''Shengda Sprint 4: US37, US38'''
        self.US37_report = List_recent_survivors(family_dict, individual_dict)
        self.US38_report = List_upcoming_birthdays(individual_dict)

    def test_US37(self):
        expect = {'@F41@': [['2020-3-27', 'NA', {'@I117@', '@I118@'}]]}
        actual = self.US37_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())


    def test_US38(self):
        expect = {'@I9@': [[datetime.datetime(1945, 4, 27, 0, 0), datetime.datetime(2020, 4, 19, 14, 41, 32, 576070)]],
                  '@I16@': [[datetime.datetime(1990, 4, 16, 0, 0), datetime.datetime(2020, 4, 19, 14, 41, 32, 576070)]],
                  '@I96@': [[datetime.datetime(1983, 4, 8, 0, 0), datetime.datetime(2020, 4, 19, 14, 41, 32, 576070)]],
                  '@I109@': [[datetime.datetime(1966, 4, 3, 0, 0), datetime.datetime(2020, 4, 19, 14, 41, 32, 591691)]],
                  '@I111@': [[datetime.datetime(1991, 4, 2, 0, 0), datetime.datetime(2020, 4, 19, 14, 41, 32, 591691)]],
                  '@I114@': [[datetime.datetime(1989, 4, 1, 0, 0), datetime.datetime(2020, 4, 19, 14, 41, 32, 591691)]]}

        actual = self.US38_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())

if __name__ == '__main__':
    unittest.main()

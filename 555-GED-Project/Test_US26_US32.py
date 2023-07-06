import unittest
from The_Real_Project03 import *

class Test_case_us26_us32(unittest.TestCase):
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
        self.US26_report = corresponding_entries(family_dict, individual_dict)
        self.US32_report = list_multiple_births(individual_dict)

    def test_US26(self):
        expect = {'@F1@': ['@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@', '@F1@'],
                  '@F2@': ['@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@', '@F2@'],
                  '@F3@': ['@F3@'], '@F4@': ['@F4@'], '@F8@': ['@F9@'], '@F9@': ['@F9@'],
                  '@F10@': ['@F11@'], '@F13@': ['@F13@', '@F13@', '@F13@'],
                  '@F17@': ['@F17@', '@F17@', '@F17@'], '@F18@': ['@F18@'],
                  '@F20@': ['@F20@'], '@F21@': ['@F21@', '@F21@'], '@F22@': ['@F22@']}
        actual = self.US26_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())


    def test_US32(self):
        expect = {'1960-1-1': ['@I1@', '@I2@', '@I6@', '@I7@', '@I8@', '@I10@', '@I11@', '@I12@', '@I13@', '@I14@', '@I16@', '@I17@', '@I18@', '@I19@', '@I22@', '@I23@', '@I42@', '@I43@'],
                       '1930-1-1': ['@I3@', '@I4@'], '1963-1-1': ['@I5@'], '1945-1-1': ['@I9@'], '1990-1-1': ['@I15@', '@I44@', '@I45@', '@I46@', '@I48@', '@I49@', '@I52@', '@I53@', '@I65@'],
                       '1840-1-1': ['@I20@'], '1850-1-1': ['@I21@'],
                       '1700-1-1': ['@I24@', '@I25@'],
                       'NA': ['@I26@', '@I27@', '@I28@', '@I29@', '@I30@', '@I31@', '@I32@', '@I33@', '@I34@', '@I35@', '@I36@', '@I37@', '@I38@', '@I39@', '@I40@', '@I41@'],
                       '1991-1-1': ['@I47@'], '1980-1-1': ['@I50@'], '1962-1-1': ['@I51@'], '1961-1-1': ['@I54@'], '1963-1-2': ['@I55@'], '2003-1-1': ['@I56@'], '2004-4-4': ['@I57@'],
                       '2005-5-3': ['@I58@'], '1960-6-6': ['@I59@', '@I72@'], '1960-3-8': ['@I60@', '@I73@'], '1994-3-2': ['@I61@'], '1959-1-1': ['@I62@'], '1969-1-3': ['@I63@'],
                       '1972-12-8': ['@I64@'], '1994-1-2': ['@I66@'], '2020-4-3': ['@I67@', '@I68@'], '1962-9-3': ['@I69@'], '1960-3-10': ['@I70@'], '1990-5-9': ['@I71@']}

        actual = self.US32_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())

if __name__ == '__main__':
    unittest.main()

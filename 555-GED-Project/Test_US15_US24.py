import unittest
from The_Real_Project03 import *

class Test_case_us15_us24(unittest.TestCase):
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
        self.US15_report = fewer_than_15_siblings(family_dict)
        self.US24_report = unique_families_by_spouses(family_dict, individual_dict)

    def test_US15(self):
        expect = {'@F1@': ['@I35@', '@I38@', '@I29@', '@I30@', '@I27@', '@I36@', '@I39@', '@I26@', '@I32@', '@I34@', '@I37@', '@I40@', '@I28@', '@I41@', '@I33@', '@I31@'],
                  '@F2@': ['@I69@', '@I51@', '@I6@', '@I17@', '@I42@', '@I59@', '@I9@', '@I22@', '@I5@', '@I12@', '@I1@', '@I50@', '@I54@', '@I10@', '@I63@', '@I72@', '@I14@', '@I8@']}
        actual = self.US15_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())


    def test_US24(self):
        expect = {'@F2@': ['NormalFather /Project555/', 'NomalMother /Project555/', '1950-1-1'],
                  '@F5@': ['US02 /Project555/', 'US02Partner /Project555/', '1950-1-1'],
                  '@F6@': ['US04 /Project555/', 'US04Partner /Project555/', '1980-1-1'],
                  '@F7@': ['US06 /Project555/', 'US06Partner /Project555/', '1980-1-1'],
                  '@F8@': ['US17 /Project555/', 'US17ChildPartner /Project555/', '1980-1-1'],
                  '@F9@': ['US17 /Project555/', 'US17Partner /Project555/', '1970-1-1'],
                  '@F10@': ['US11 /Project555/', 'US11SecPartner /Project555/', '1980-1-1'],
                  '@F11@': ['US11 /Project555/', 'US11Partner /Project555/', '1979-1-1'],
                  '@F12@': ['US05 /Project555/', 'US05Partner /Project555/', '1980-1-1'],
                  '@F14@': ['US24 /Project555/', 'US24Partner /Project555/', '2010-1-1'],
                  '@F15@': ['US242 /Project555/', 'US24Partner /Project555/', '2010-1-1'],
                  '@F16@': ['US10 /Project555/', 'US10Partner /Project555/', '2000-1-1'],
                  '@F18@': ['US34Partner /Project555/', 'US34 /Project555/', '1989-8-8']}

        actual = self.US24_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())

if __name__ == '__main__':
    unittest.main()

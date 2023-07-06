#!/usr/bin/env python
# coding: utf-8

# In[4]:


import unittest
from US11And12 import *

class Test_case_us11_us12(unittest.TestCase):
    def setUp(self):
        file_path = "Hercule-Poirot.ged"
        with open(file_path, 'r') as file:
            unfiltered_file = file.readlines()
            '''filter the file so only valid lines are returned'''
            filtered_file = filter_file(unfiltered_file)
            '''draw the skeleton of Individual and Family and assign each field with NA'''
            individual = Individual()
            family = Family()

            """ Add User Stories here"""

            draw_skeleton(filtered_file, individual, family)
            fill_skeleton(filtered_file, individual, family)

            ''' Test case of us 11 and us 12 '''
            self.US11_report = us_11(individual, family)
            self.US12_report = us_12(individual, family)

    def test_US11(self):
        expect = {'@I1@': True, '@I5@': True, '@I2@': True, '@I3@': True, '@I8@': True, '@I7@': False, '@I9@': True}
        actual = self.US11_report
        self.assertEqual(actual, expect)

    def test_US12(self):
        expect = {'@F1@': True, '@F3@': True, '@F4@': True, '@F5@': True}
        actual = self.US12_report
        self.assertEqual(actual, expect)

if __name__ == '__main__':
        unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity = 2)


# In[ ]:





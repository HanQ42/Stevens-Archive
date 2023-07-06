#!/usr/bin/env python
# coding: utf-8

# In[2]:


import unittest
import The_Real_Project03

class Test_Sprint_4(unittest.TestCase):
    '''Test Cases for Sprint 4, US20 + US21'''
    '''US20 US21 Test Case'''
    def test(self):
        main = The_Real_Project03.main()
        individual_dict = main[0]
        family_dict = main[1]
        actual = main[2]
        expected = ['ERROR: US20: @I44@ should not marry his aunt or uncle @I45@',
                    'ERROR: US20: @I44@ should not marry his aunt or uncle @I45@',
                    'ERROR: US20: @I46@ should not marry his aunt or uncle @I47@',
                    'ERROR: US20: @I46@ should not marry his aunt or uncle @I47@',
                    'ERROR: US20: @I48@ should not marry his aunt or uncle @I49@',
                    'ERROR: US20: @I48@ should not marry his aunt or uncle @I49@',
                    'ERROR: US21: @I54@ gender is not correct',
                    'ERROR: US21: @I56@ gender is not correct']
        self.assertEqual(expected, actual)

if __name__ == '__main__':
        unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity = 2)


# In[ ]:





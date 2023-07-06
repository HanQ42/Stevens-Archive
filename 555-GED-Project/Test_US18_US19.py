import unittest
import The_Real_Project03

class Test_Sprint_3(unittest.TestCase):
    '''Test Cases for Sprint 3, US18 + US19'''
    '''US18 US19 Test Case'''
    def test(self):
        main = The_Real_Project03.main()
        individual_dict = main[0]
        family_dict = main[1]
        actual = main[2]
        expected = ['ERROR: US18: @I102@ should not marry with sibling @I103@',
                    'ERROR: US18: @I103@ should not marry with sibling @I102@',
                    'ERROR: US19: @I105@ should not marry his cousin @I106@',
                    'ERROR: US19: @I106@ should not marry his cousin @I105@']
        self.assertEqual(expected, actual)

if __name__ == '__main__':
        unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity = 2)


# In[ ]:

import unittest
import Project03CornerStone
class TestUS_01_02(unittest.TestCase):
    def test_us_01_02(self):
        expected = [{'@I1@': {'NAME': 'James /Corden/', 'SEX': 'M', 'BIRT': '1980-1-1', 'DEAT': 'NA', 'FAMC': '@F2@', 'FAMS': '@F1@'},
                     '@I2@': {'NAME': 'Abby /Corden/', 'SEX': 'F', 'BIRT': '1981-1-1', 'DEAT': 'NA', 'FAMC': 'NA', 'FAMS': '@F1@'},
                     '@I3@': {'NAME': 'Roman /Corden/', 'SEX': 'M', 'BIRT': 'NA', 'DEAT': 'NA', 'FAMC': 'NA', 'FAMS': '@F2@'},
                     '@I4@': {'NAME': 'Juliet /Corden/', 'SEX': 'F', 'BIRT': '1950-1-1', 'DEAT': 'NA', 'FAMC': 'NA', 'FAMS': '@F2@'}},
                    {'@F1@': {'MARR': 'NA', 'HUSB': ['@I1@', 'James /Corden/'], 'WIFE': ['@I2@', 'Abby /Corden/'], 'CHIL': set(), 'DIV': 'NA'},
                     '@F2@': {'MARR': '1999-1-1', 'HUSB': ['@I3@', 'Roman /Corden/'], 'WIFE': ['@I4@', 'Juliet /Corden/'], 'CHIL': {'@I1@'}, 'DIV': 'NA'}}]
        return_value = Project03CornerStone.main()
        self.assertEqual(return_value, expected)

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
import The_Real_Project03
import unittest

class TestUS23_US28(unittest.TestCase):
    '''Set up'''
    file_path = '555Project(updates-often).ged'
    individuals = {}
    families = {}
    with open(file_path, 'r') as file:
        unfiltered_file = file.readlines()
        The_Real_Project03.filter_file(unfiltered_file, individuals, families)
    individual_dict = {}
    family_dict = {}
    The_Real_Project03.raw_individuals_to_structured_dict(individuals, individual_dict)
    The_Real_Project03.raw_families_to_structured_dict(families, family_dict)

    def test_US23(self):
        error_report = The_Real_Project03.main()
        exp1 = "ERROR: US23: Individuals @I65@ and @I66@ have the same name US25 SP3 and the same birthday 1955-1-1."
        exp2 = "ERROR: US23: Individuals @I103@ and @I104@ have the same name US23 SP4 and the same birthday 1990-1-1."
        self.assertIn(exp1, error_report.error_list)
        self.assertIn(exp2, error_report.error_list)

    def test_US28(self):
        exp = ['US28: Siblings in family @F2@ ordered by decreasing age is @I9@ @I10@ @I12@ @I14@ @I17@ @I1@ @I22@ @I42@ @I53@ @I6@ @I8@ @I51@ @I50@ @I5@',
               'US28: Siblings in family @F18@ ordered by decreasing age is @I65@ @I66@ @I54@ @I57@ @I58@ @I67@ @I71@ @I74@ @I95@ @I73@',
               'US28: Siblings in family @F27@ ordered by decreasing age is @I80@ @I84@ @I86@ @I92@ @I93@ @I79@']
        output = The_Real_Project03.order_siblings_by_age(self.individual_dict, self.family_dict)
        self.assertEqual(exp, output)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
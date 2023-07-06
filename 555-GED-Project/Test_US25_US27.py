import unittest
import The_Real_Project03


class TestUS25(unittest.TestCase):

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

    def test_US25(self):
        error_report = The_Real_Project03.main()
        exp1 = "Error: US25: Family @F2@ has multiple children with first name US23."

        self.assertIn(exp1, error_report.error_list)

    def test_US27(self):
        error_report = The_Real_Project03.main()
        exp1 = "INDIVIDUAL: US27: Individual ID: @I1@ is 60 years old."

        self.assertIn(exp1, error_report.error_list)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)



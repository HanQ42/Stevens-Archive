import The_Real_Project03
import unittest


class TestUS29US30(unittest.TestCase):

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

    def test_US29(self):
        error_report = The_Real_Project03.main()
        exp1 = "INDIVIDUAL: US29: Individual ID: @I8@, with death date 1959-01-01 00:00:00, is a person who has died."

        self.assertIn(exp1, error_report.error_list)

    def test_US30(self):
        error_report = The_Real_Project03.main()
        exp = "FAMILY: US30: Husband:@I12@ and Wife:@I13@ are married and living. "

        self.assertIn(exp, error_report.error_list)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

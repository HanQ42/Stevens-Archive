import unittest
import The_Real_Project03

class TestSprint03(unittest.TestCase):
    error_repo = The_Real_Project03.main()
    def test_us09_us16(self):
        errors = self.error_repo.error_list

        expect_error1 = "ERROR: US09: Individual @I54@ has a birthday happens 9 months after father's death."
        expect_error2 = "ERROR: US16: Family @F18@ has a male child has a last name WrongLastName instead of SP3."

        self.assertIn(expect_error1, errors)
        self.assertIn(expect_error2, errors)

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

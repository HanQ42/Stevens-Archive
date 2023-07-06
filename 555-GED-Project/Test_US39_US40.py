import unittest
from The_Real_Project03 import *
import datetime

class Test_case_us39_us40(unittest.TestCase):
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

        '''Hanqing Sprint 1: US01, US02'''
        dates_before_current_date(individual_dict, family_dict)  # US01
        birth_before_marriage(individual_dict, family_dict)  # US02
        '''Hanqing Sprint 2: US03, US08'''
        birth_before_death(individual_dict)  # US03
        birth_before_marriage_of_parents(individual_dict, family_dict)  # US08
        '''Hanqing Sprint 3: US09, US16'''
        birth_before_parents_death(individual_dict, family_dict)  # US09
        male_last_names(individual_dict, family_dict)  # US16
        '''Hanqing Sprint 4: US23, US28'''
        unique_name_and_birth_date(individual_dict)  # US23
        order_siblings_by_age(individual_dict, family_dict)  # US28

        '''Jigar Sprint 1: US04, US06'''
        marriage_before_divorce(family_dict)  # US04
        divorce_before_death(family_dict, individual_dict)  # US06
        '''Jigar Sprint 2: US17, US22'''
        no_marriage_to_children(family_dict, individual_dict)  # US17
        unique_ids(unfiltered_file)  # US22
        '''Jigar Sprint 3: US25, US27'''
        unique_first_name(family_dict, individual_dict)  # US25
        include_individual_ages(individual_dict)  # US27

        '''Shengda Sprint 1: US05, US07'''
        marriage_before_death(family_dict, individual_dict)  # US05
        less_than_150_years_old(individual_dict)  # US07
        '''Shengda Sprint 2: US10, US31'''
        marriage_after_14(family_dict, individual_dict)  # US10
        list_living_single(family_dict, individual_dict)  # US31
        '''shengda Sprint 3: US35, US36'''
        List_recent_births(individual_dict)
        List_recent_deaths(individual_dict)
        '''shengda Sprint 4: US37, US38'''
        List_recent_survivors(family_dict, individual_dict)
        List_upcoming_birthdays(individual_dict)

        '''Haoran Sprint 1: US11, US12'''
        no_bigamy(family_dict, individual_dict)  # US11
        parents_not_too_old(family_dict, individual_dict)  # US12
        '''Haoran Sprint 2: US13, US14'''
        siblings_spacing(family_dict, individual_dict)  # US13
        mutiple_birth(family_dict, individual_dict)  # US14
        '''Haoran Sprint 3: US18, US19'''
        sibling_not_marry(family_dict, individual_dict)  # US18
        first_cousin_not_marry(family_dict, individual_dict)  # US19

        '''Xiangyu Sprint 1: US15, US24'''
        fewer_than_15_siblings(family_dict)  # US15
        unique_families_by_spouses(family_dict, individual_dict)  # US24
        '''Xiangyu Sprint 2: US26, US32'''
        corresponding_entries(family_dict, individual_dict)  # US26
        list_multiple_births(individual_dict)  # US32
        '''Xiangyu Sprint 3: US33, US34'''
        list_orphans(family_dict, individual_dict)
        list_large_age_differences(family_dict, individual_dict)
        '''Xiangyu Sprint 4: US39, US40'''
        self.US39_report = List_upcoming_anniversaries(family_dict)
        #self.US40_report = Include_input_line_numbers(ErrorCollector.error_list)

    def test_US39(self):
        expect = {'@F8@': [['@I14@', '@I15@']], '@F13@': [['@I42@', '@I43@']]}
        actual = self.US39_report
        for key in expect.keys():
            self.assertEqual(actual[key].sort(), expect[key].sort())


    def test_US40(self):

        expect = {'01': 10, '02': 8,'03': 2, '04': 4, '05': 4, '06': 4, '07': 10, '08': 2, '09': 28, '10': 14,
                  '11': 2, '12': 16, '13': 66, '14': 32, '15': 4,'16': 2, '17': 2, '18': 6, '19': 10, '22': 2, '23': 4, '24': 2, '25': 4,'26': 4, '28': 14, '31': 34, '32': 14, '33': 4, '34': 4,'35': 6, '36': 4,'37': 2,'38': 12,'39': 4}

        actual = Include_input_line_numbers(ErrorCollector.error_list)
        self.assertEqual(actual, expect)

if __name__ == '__main__':
    unittest.main()

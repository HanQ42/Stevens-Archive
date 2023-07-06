#!/usr/bin/env python
# coding: utf-8

# In[37]:


"""The Brand New Project 03 With Better Data Structure And Better Visual"""
"""Please Make Code As Minimalist As Possible"""
from prettytable import PrettyTable
from datetime import date
import datetime

"""Useful Functions And Classes"""
'''Individual Class'''
class Individual:
    def __init__(self, name, sex, birt, deat, famc, fams):
        self.name = name
        self.sex = sex
        self.birt = birt
        self.deat = deat
        self.famc = famc
        self.fams = fams

'''Family Class'''
class Family:
    def __init__(self, marr, husb, wife, chil, div):
        self.marr = marr
        self.husb = husb
        self.wife = wife
        self.chil = chil
        self.div = div

'''Date Class'''
class DateObject:
    def __init__(self, string_date):
        self.string_date = string_date
        self.year = 'NA'
        self.month = 'NA'
        self.day = 'NA'
        self.parse_string_date(string_date)

    def parse_string_date(self, string_date):
        '''1 Feb 1990 comes in'''
        if string_date == 'NA':
            self.year, self.month, self.day = 'NA', 'NA', 'NA'
        else:
            string_date_list = string_date.split(' ')
            month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
            self.year = string_date_list[2]
            self.month = month_dict[string_date_list[1]]
            self.day = string_date_list[0]

    def snake_year_month_day(self):
        '''1990-2-1 comes out'''
        if self.year == 'NA' or self.month == 'NA' or self.day == 'NA':
            return 'NA'
        else:
            return f'{self.year}-{self.month}-{self.day}'

    def setNA(self):
        self.year, self.month, self.day = 'NA', 'NA', 'NA'

'''Age Calculator'''
def age_calculator(later_date, earlier_date):
    '''DateObjects, first parameter is for a later date, the second is for earlier'''
    def age_difference(later_date, earlier_date):
        y_l, m_l, d_l = later_date.year, later_date.month, later_date.day
        y_e, m_e, d_e = earlier_date.year, earlier_date.month, earlier_date.day
        return int(y_l) - int(y_e) - ((int(m_l), int(d_l)) < (int(m_e), int(d_e)))
    if isinstance(later_date, date):
        if earlier_date.snake_year_month_day() == 'NA':
            return 'NA'
        else:
            return age_difference(later_date, earlier_date)
    else:
        if later_date.snake_year_month_day() == 'NA' or earlier_date.snake_year_month_day() == 'NA':
            return 'NA'
        else:
            return age_difference(later_date, earlier_date)
'''Days Calculator'''
def days_calculator(later_date, earlier_date):
    y_l, m_l, d_l = later_date.year, later_date.month, later_date.day
    y_e, m_e, d_e = earlier_date.year, earlier_date.month, earlier_date.day
    def days_difference():
        delta = date(int(y_l), int(m_l), int(d_l)) - date(int(y_e), int(m_e), int(d_e))
        return delta.days
    if isinstance(later_date, date):
        if earlier_date.snake_year_month_day() == 'NA':
            return 'NA'
        else:
            return days_difference()
    else:
        if later_date.snake_year_month_day() == 'NA' or earlier_date.snake_year_month_day() == 'NA':
            return 'NA'
        else:
            return days_difference()

'''Individual & Family Dictionary Constructor'''
def raw_individuals_to_structured_dict(raw_individuals, individual_dict):
    for key, value in raw_individuals.items():
        name, sex, birt, deat, famc, fams = 'NA', 'NA', DateObject('NA'), DateObject('NA'), 'NA', 'NA'
        for entry in range(len(value)):
            if value[entry][0] == '1' and value[entry][1] == 'NAME':
                name = value[entry][2]
            elif value[entry][0] == '1' and value[entry][1] == 'SEX':
                sex = value[entry][2]
            elif value[entry][0] == '1' and value[entry][1] == 'BIRT':
                if value[entry + 1][0] == '2' and value[entry + 1][1] == 'DATE':
                    birt = DateObject(value[entry + 1][2])
            elif value[entry][0] == '1' and value[entry][1] == 'DEAT':
                if value[entry + 1][0] == '2' and value[entry + 1][1] == 'DATE':
                    deat = DateObject(value[entry + 1][2])
            elif value[entry][0] == '1' and value[entry][1] == 'FAMC':
                famc = value[entry][2]
            elif value[entry][0] == '1' and value[entry][1] == 'FAMS':
                fams = value[entry][2]
        individual_dict[key] = Individual(name, sex, birt, deat, famc, fams)

def raw_families_to_structured_dict(raw_families, family_dict):
    for key, value in raw_families.items():
        marr, husb, wife, chil, div = DateObject('NA'), 'NA', 'NA', set(), DateObject('NA')
        for entry in range(len(value)):
            if value[entry][0] == '1' and value[entry][1] == 'MARR':
                if value[entry + 1][0] == '2' and value[entry + 1][1] == 'DATE':
                    marr = DateObject(value[entry + 1][2])
            elif value[entry][0] == '1' and value[entry][1] == 'HUSB':
                husb = value[entry][2]
            elif value[entry][0] == '1' and value[entry][1] == 'WIFE':
                wife = value[entry][2]
            elif value[entry][0] == '1' and value[entry][1] == 'CHIL':
                chil.add(value[entry][2])
            elif value[entry][0] == '1' and value[entry][1] == 'DIV':
                if value[entry + 1][0] == '2' and value[entry + 1][1] == 'DATE':
                    div = DateObject(value[entry + 1][2])
        if len(chil) == 0:
            chil = 'NA'
        family_dict[key] = Family(marr, husb, wife, chil, div)

'''Pretty Tables'''
def draw_individual_prettytable(individual_dict):
    '''this pretty table uses age_calculator'''
    pt = PrettyTable()
    pt.field_names = ["ID", "Name", "Gender", "Birthday", "Age", "Alive", "Death", "Child", "Spouse"]
    for id, individual in individual_dict.items():
        alive = True
        death_date = 'NA'
        if individual.birt.snake_year_month_day() == 'NA':
            alive = 'NA'
        if individual.deat.snake_year_month_day() != 'NA':
            alive = False
            death_date = individual.deat.snake_year_month_day()
        pt.add_row([id, individual.name, individual.sex, individual.birt.snake_year_month_day(), age_calculator(date.today(), individual.birt), alive, death_date, individual.famc, individual.fams])
    print(pt)

def draw_family_prettytable(family_dict, individual_dict):
    pt = PrettyTable()
    pt.field_names = ["ID", "Married", "Divorced", "Husband ID", "Husband Name", "Wife ID", "Wife Name", "Children"]
    for id, family in family_dict.items():
        '''marr, husb, wife, chil, div'''
        pt.add_row([id, family.marr.snake_year_month_day(), family.div.snake_year_month_day(), family.husb, individual_dict[family.husb].name if family.husb != 'NA' else 'NA', family.wife, individual_dict[family.wife].name if family.wife != 'NA' else 'NA', family.chil])
    print(pt)

'''File Filter'''
def filter_file(file_to_filter, individuals, families):
    # valid_tags_for_0 = ['HEAD', 'TRLR', 'NOTE']
    valid_tags_for_1 = ['NAME', 'SEX', 'BIRT', 'DEAT', 'FAMC', 'FAMS', 'MARR', 'HUSB', 'WIFE', 'CHIL', 'DIV']
    valid_tags_for_2 = ['DATE']

    individual_or_family_flag = ''
    individual_id = ''
    family_id = ''
    for line in file_to_filter:
        if line == '':
            continue
        else:
            line = line.strip('\n')
            line_split = line.split(' ')
            if line_split[0] == '0':
                if 'INDI' in line_split and line_split[1] != 'INDI':
                    individual_or_family_flag = 'individual'
                    individual_id = line_split[1]
                    individuals[individual_id] = []
                elif 'FAM' in line_split and line_split[1] != 'FAM':
                    individual_or_family_flag = 'family'
                    family_id = line_split[1]
                    families[family_id] = []
            elif line_split[0] == '1':
                if line_split[1] in valid_tags_for_1:
                    if individual_or_family_flag == 'individual':
                        individuals[individual_id].append(line.split(' ', 2))
                    elif individual_or_family_flag == 'family':
                        families[family_id].append(line.split(' ', 2))
            elif line_split[0] == '2':
                if line_split[1] in valid_tags_for_2:
                    if individual_or_family_flag == 'individual':
                        individuals[individual_id].append(line.split(' ', 2))
                    elif individual_or_family_flag == 'family':
                        families[family_id].append(line.split(' ', 2))

'''Error Collector'''
class ErrorCollector:
    error_list = []

"""Hanqing's Code Goes Here"""
'''Sprint 1'''
'''User Story 01: Dates before current date'''
def dates_before_current_date(individual_dict, family_dict):
    '''This function uses age_calculator, any dates in the future becomes NA'''
    # There are 4 fields using date: birt, deat, marr, div
    for key, value in individual_dict.items():
        # birt
        if value.birt.snake_year_month_day() != 'NA':
            if isinstance(age_calculator(date.today(), value.birt), int) and age_calculator(date.today(), value.birt) <= 0:
                ErrorCollector.error_list.append(f"ERROR: US01: Individual {key} has a birthday {value.birt.snake_year_month_day()} occurs in the future. Birthday was set to NA.")
                # value.birt.setNA()
        #deat
        if value.deat.snake_year_month_day() != 'NA':
            if isinstance(age_calculator(date.today(), value.deat), int) and age_calculator(date.today(), value.deat) <= 0:
                ErrorCollector.error_list.append(f"ERROR: US01: Individual {key} has a dearh date {value.birt.snake_year_month_day()} occurs in the future. Death date was set to NA.")
                # value.deat.setNA()
    for key, value in family_dict.items():
        # marr
        if value.marr.snake_year_month_day() != 'NA':
            if age_calculator(date.today(), value.marr) <= 0:
                ErrorCollector.error_list.append(f"ERROR: US01: Individual {key} has a wedding date {value.marr.snake_year_month_day()} occurs in the future. Wedding date was set to NA.")
                # value.marr.setNA()
        # div
        if value.div.snake_year_month_day() != 'NA':
            if age_calculator(date.today(), value.div) <= 0:
                ErrorCollector.error_list.append(f"ERROR: US01: Individual {key} has a divorce date {value.div.snake_year_month_day()} occurs in the future. Divorce date was set to NA.")
                # value.div.setNA()

'''User Story 02: Birth before marriage'''
def birth_before_marriage(individual_dict, family_dict):
    '''Any wedding date before birthday becomes NA'''
    for key, value in individual_dict.items():
        if value.fams != 'NA':
            if value.birt.snake_year_month_day() != 'NA' and family_dict[value.fams].marr.snake_year_month_day() != 'NA':
                if age_calculator(family_dict[value.fams].marr, value.birt) < 0:
                    ErrorCollector.error_list.append(f"ERROR: US02: Individual {key} has a wedding date {family_dict[value.fams].marr.snake_year_month_day()} occurs before birthday {value.birt.snake_year_month_day()}. Wedding date was set to NA.")
                    # family_dict[value.fams].marr.setNA()

'''Sprint 2'''
'''User Story 03: Birth Before Death'''
def birth_before_death(individual_dict):
    for id, value in individual_dict.items():
        if value.birt.snake_year_month_day() != 'NA' and value.deat.snake_year_month_day() != 'NA':
            if age_calculator(value.deat, value.birt) <= 0:
                ErrorCollector.error_list.append(f"ERROR: US03: Individual {id} has a birthday {value.birt.snake_year_month_day()} that occurred after death date {value.deat.snake_year_month_day()}. Birthday was set to NA.")
                # value.birt.setNA()

'''User Story 08: Birth Before Marriage Of Parents'''
def birth_before_marriage_of_parents(individual_dict, family_dict):
    for id, value in individual_dict.items():
        if value.famc != 'NA':
            if family_dict[value.famc].marr != 'NA':
                if age_calculator(value.birt, family_dict[value.famc].marr) != 'NA' and age_calculator(value.birt, family_dict[value.famc].marr) < 0:
                    ErrorCollector.error_list.append(f"ERROR: US08: Individual {id} has a birth date {value.birt.snake_year_month_day()} that's earlier than parents' wedding date {family_dict[value.famc].marr.snake_year_month_day()}. Birthday was set to NA.")
                    # value.birt.setNA()

'''Sprint 3'''
'''US09 Birth before Parents' Death'''
def birth_before_parents_death(individual_dict, family_dict):
    for id, individual in individual_dict.items():
        if individual.famc != 'NA':
            father = family_dict[individual.famc].husb
            mother = family_dict[individual.famc].wife
            if father != 'NA' and mother != 'NA':
                father_death = individual_dict[father].deat
                mother_death = individual_dict[mother].deat
                if father_death.snake_year_month_day() != 'NA' and days_calculator(individual.birt, father_death) != 'NA':
                    if days_calculator(individual.birt, father_death) > 274:
                        ErrorCollector.error_list.append(f"ERROR: US09: Individual {id} has a birthday happens 9 months after father's death.")
                if mother_death.snake_year_month_day() != 'NA' and days_calculator(individual.birt, father_death) != 'NA':
                    if days_calculator(individual.birt, mother_death) > 0:
                        ErrorCollector.error_list.append(f"ERROR: US09: Individual {id} has a birthday after mother's death.")

'''US16 Male Last Names'''
def male_last_names(individual_dict, family_dict):
    for id, family in family_dict.items():
        husb_lastname = individual_dict[family.husb].name.replace('/', '').split(' ')[1]
        if family.chil != 'NA':
            for child in family.chil:
                if child in individual_dict:
                    if individual_dict[child].sex == 'M':
                        child_lastname = individual_dict[child].name.replace('/', '').split(' ')[1]
                        if child_lastname != husb_lastname:
                            ErrorCollector.error_list.append(f"ERROR: US16: Family {id} has a male child has a last name {child_lastname} instead of {husb_lastname}.")

'''Sprint 4'''
'''US23 Unique Name and Birth Date'''
def unique_name_and_birth_date(individual_dict):
    name_id_dict = {}
    for id, individual in individual_dict.items():
        name = individual.name.replace('/', '')
        if name not in name_id_dict:
            name_id_dict[name] = [id]
        else:
            name_id_dict[name].append(id)
    for name, id_list in name_id_dict.items():
        if len(id_list) > 1:
            birthdays = {}
            for id in id_list:
                if individual_dict[id].birt.snake_year_month_day() != 'NA':
                    if individual_dict[id].birt.snake_year_month_day() not in birthdays:
                        birthdays[individual_dict[id].birt.snake_year_month_day()] = [id]
                    else:
                        birthdays[individual_dict[id].birt.snake_year_month_day()].append(id)
            for birthday, id_list in birthdays.items():
                if len(id_list) > 1:
                    ErrorCollector.error_list.append(f"ERROR: US23: Individuals {' and '.join(id_list)} have the same name {name} and the same birthday {birthday}.")

'''US28 Order Siblings By Age'''
def order_siblings_by_age(individual_dict, family_dict):
    return_list = []
    for id, family in family_dict.items():
        if family.chil != 'NA' and len(family.chil) > 1:
            child_age_dict = {}
            for child in family.chil:
                if child in individual_dict:
                    child_age_dict[child] = age_calculator(date.today(), individual_dict[child].birt)

            copy_of_cad = child_age_dict.copy()
            for child, age in child_age_dict.items():
                if age == 'NA':
                    del copy_of_cad[child]
            sorted_child_age_list = sorted(copy_of_cad.items(), key=lambda x: x[1])
            sorted_child_age_list.reverse()

            age_sib_order_dict = {}

            for item in sorted_child_age_list:
                if item[1] not in age_sib_order_dict:
                    age_sib_order_dict[item[1]] = [item[0]]
                else:
                    age_sib_order_dict[item[1]].append(item[0])
            for age, sib in age_sib_order_dict.items():
                sib.sort()
            temp_return_list = []
            if len(age_sib_order_dict) > 1:
                for age, sibs in age_sib_order_dict.items():
                    for sib in sibs:
                        temp_return_list.append(sib)
            '''List siblings in the family by decreasing age'''
            if len(temp_return_list) > 1:
                #print(f"US28: Siblings in family {id} ordered by decreasing age is {' '.join(temp_return_list)}")
                ErrorCollector.error_list.append(f"US28: Siblings in family {id} ordered by decreasing age is {' '.join(temp_return_list)}")
                return_list.append(f"US28: Siblings in family {id} ordered by decreasing age is {' '.join(temp_return_list)}")
    return return_list

"""Jigar's Code Goes Here"""
'''Sprint 1'''
'''User Story 04: Marriage before divorce'''
def marriage_before_divorce(family_dict):
    for key, value in family_dict.items():
        if value.marr.snake_year_month_day() != 'NA' and value.div.snake_year_month_day() != 'NA':
            if age_calculator(value.div, value.marr) <= 0:
                wedding_date = value.marr.snake_year_month_day()
                divorce_date = value.div.snake_year_month_day()
                ErrorCollector.error_list.append(f"ERROR: US04: Family {key} has a divorce date {divorce_date} occurs before wedding date {wedding_date}. Divorce date was set to NA.")
                value.div.setNA()

'''User Story 06: Divorce before death'''
def divorce_before_death(family_dict, individual_dict):
    for key_family, value_family in family_dict.items():
        divorce_date = value_family.div
        if divorce_date.snake_year_month_day() != 'NA':
            # husband death
            husband_death = individual_dict[value_family.husb].deat
            if husband_death.snake_year_month_day() != 'NA':
                if age_calculator(husband_death, divorce_date) < 0:
                    ErrorCollector.error_list.append(f"ERROR: US06: Family {key_family} has a divorce date {divorce_date.snake_year_month_day()} occurs after husband's {value_family.husb} death at {husband_death.snake_year_month_day()}. Divorce date was set to NA.")
                    divorce_date.setNA()
            # wife death
            wife_death = individual_dict[value_family.wife].deat
            if wife_death.snake_year_month_day() != 'NA':
                if age_calculator(wife_death, divorce_date) < 0:
                    ErrorCollector.error_list.append(f"ERROR: US06: Family {key_family} has a divorce date {divorce_date.snake_year_month_day()} occurs after wife's {value_family.wife} death at {wife_death.snake_year_month_day()}. Divorce date was set to NA.")
                    divorce_date.setNA()

'''Sprint 2'''
'''User Story 17: No marriages to children'''
def no_marriage_to_children(family_dict, individual_dict):
    for key, individual in individual_dict.items():
        if individual.famc != 'NA' and individual.fams != 'NA':
            husband = family_dict[individual.fams].husb
            wife = family_dict[individual.fams].wife
            father = family_dict[individual.famc].husb
            mother = family_dict[individual.famc].wife
            if husband == father:
                ErrorCollector.error_list.append(f"ERROR: US17: Individual {key} is married to her father {father} which is illegal. Family eliminated.")
                del family_dict[individual.fams]
                individual_dict[husband].fams = 'NA'
                individual.fams = 'NA'
            elif wife == mother:
                ErrorCollector.error_list.append(f"ERROR: US17: Individual {key} is married to his mother {mother} which is illegal. Family eliminated.")
                del family_dict[individual.fams]
                individual.fams = 'NA'
                individual_dict[wife].fams = 'NA'

'''User Story 22: Unique IDs'''
def unique_ids(unfiltered_file):
    individuals = []
    families = []
    for line in unfiltered_file:
        if line == '':
            continue
        else:
            line = line.strip('\n')
            line_split = line.split(' ')
            if line_split[0] == '0':
                if 'INDI' in line_split and line_split[1] != 'INDI':
                    id = line_split[1]
                    if id in individuals:
                        ErrorCollector.error_list.append(f"ERROR: US22: Individual {id} already exists and will override previous data")
                    else:
                        individuals.append(id)
                elif 'FAM' in line_split and line_split[1] != 'FAM':
                    id = line_split[1]
                    if id in families:
                        ErrorCollector.error_list.append(f"ERROR: US22: Family {id} already exists and will override previous data")
                    else:
                        families.append(id)
                        
'''Sprint 3'''
'''User Story 25: Unique first names in families'''
def unique_first_name(family_dict, individual_dict):
    for fam_id, family in family_dict.items():
        if family.chil != 'NA':
            first_names = []
            for family_child in family.chil:
                if family_child in individual_dict:
                    first_name = individual_dict[family_child].name.split(' ')[0]
                    if first_name not in first_names:
                        first_names.append(first_name)
                    else:
                        ErrorCollector.error_list.append(f"Error: US25: Family {fam_id} has multiple children with first name {first_name}.")

    
'''User Story 27: Unique first names in families'''
def include_individual_ages(individual_dict):
    id_age_list = []
    for id, individual in individual_dict.items():
        ErrorCollector.error_list.append(f"INDIVIDUAL: US27: Individual ID: {id} is {age_calculator(date.today(), individual.birt)} years old.")
   

                                   
'''Sprint 4'''
'''User Story 29: List deceased'''


def list_deceased(individual_dict):
    deceased_list = {}

    for indi, individual in individual_dict.items():
        indi_death_date = individual.deat.snake_year_month_day()
        if indi_death_date != 'NA':
            string_date_list = indi_death_date.split('-')
            year = int(string_date_list[0])
            month = int(string_date_list[1])
            day = int(string_date_list[2])
            current_time = datetime.datetime.now()
            death_date = datetime.datetime(year, month, day)
            difference_days = (current_time - death_date).days
            if difference_days <= 999999999999:
                deceased_list.setdefault(indi, []).extend([death_date, current_time])

    for id, date in deceased_list.items():
        ErrorCollector.error_list.append(f"INDIVIDUAL: US29: Individual ID: {id}, with death date {date[0]}, "
                                         f"is a person who has died.")

    return deceased_list


'''User Story 30: List living married'''


def list_living_married(family_dict, individual_dict):
    living_married = {}

    for fam, family in family_dict.items():
        hid = family.husb
        wid = family.wife
        children = family.chil
        if hid != 'NA' and wid != 'NA':
            husband_death_date = individual_dict[hid].deat.snake_year_month_day()
            wife_death_date = individual_dict[wid].deat.snake_year_month_day()
            if husband_death_date != 'NA' and wife_death_date == 'NA':
                deat_datetime = datetime.datetime.strptime(husband_death_date, "%Y-%m-%d")
                current_datetime = datetime.datetime.now()
                dif_datetime = current_datetime - deat_datetime
                if dif_datetime.days <= 9999999999:
                    living_married.setdefault(fam, []).append([husband_death_date, wife_death_date, children])
                    ErrorCollector.error_list.append(f"FAMILY: US30: Husband:{hid} and Wife:{wid} are married and living. ")

    return living_married
                                   
    
"""Haoran's Code Goes Here"""
'''Sprint 1'''
'''User Story 11: No Bigamy'''
def no_bigamy(family_dict, individual_dict):
    individual_family_dict = {}
    for key, value in family_dict.items():
        if value.husb not in individual_family_dict:
            individual_family_dict[value.husb] = [key]
        else:
            individual_family_dict[value.husb].append(key)
        if value.wife not in individual_family_dict:
            individual_family_dict[value.wife] = [key]
        else:
            individual_family_dict[value.wife].append(key)
    for key, value in individual_family_dict.items():
        if len(value) > 1:
            # if one individual has more than 1 family.
            # correct their family in individual_dict and delete the second family from family_dict
            # set fams of the unlucky bast**d to NA
            ErrorCollector.error_list.append(f"ERROR: US11: {key} committed bigamy and the second family {value[1]} will be eliminated.")
            
'''User Story 12: Parents Not Too Old'''
def parents_not_too_old(family_dict, individual_dict):
    for id, individual in individual_dict.items():
        if individual.famc != 'NA':
            dad = family_dict[individual.famc].husb
            mom = family_dict[individual.famc].wife
            how_much_older_is_dad = 'NA'
            how_much_older_is_mom = 'NA'
            if dad != 'NA':
                how_much_older_is_dad = age_calculator(individual.birt, individual_dict[dad].birt)
            if mom != 'NA':
                how_much_older_is_mom = age_calculator(individual.birt, individual_dict[mom].birt)
            if how_much_older_is_dad != 'NA' and how_much_older_is_dad >= 80:
                ErrorCollector.error_list.append(f"ERROR: US12: {id} has a father who is {how_much_older_is_dad} older which is more than 80 and birthday is set to NA")
                individual_dict[dad].birt.setNA()
            if how_much_older_is_mom != 'NA' and how_much_older_is_mom >= 60:
                ErrorCollector.error_list.append(f"ERROR: US12: {id} has a father who is {how_much_older_is_mom} older which is more than 60 and birthday is set to NA")
                individual_dict[mom].birt.setNA()

"""Sprint 2"""
"""User Story 13: Siblings spacing"""
def siblings_spacing(family_dict, individual_dict):
    for id, individual in individual_dict.items():
        if individual.famc != 'NA':
            siblings = family_dict[individual.famc].chil
            if len(siblings) != 1:
                for child in siblings:
                    if child in individual_dict:
                        if individual_dict[child].birt.year != 'NA' and individual_dict[child].birt.month != 'NA' and individual_dict[child].birt.day != 'NA' and individual.birt.year != 'NA' and individual.birt.month != 'NA' and individual.birt.day != 'NA':
                            sib_age = date(int(float(individual_dict[child].birt.year)), int(float(individual_dict[child].birt.month)),
                                           int(float(individual_dict[child].birt.day)))
                            self_age = date(int(float(individual.birt.year)), int(float(individual.birt.month)),
                                            int(float(individual.birt.day)))
                            difference = abs(self_age - sib_age).days
                            '''8 months is 240 days '''
                            #print(difference)
                            if difference > 1 and difference < 240:
                                ErrorCollector.error_list.append(f"ERROR: US13: {id} has a sibling whose birth date is too close")


"""User story 14: Mutiple births <= 5"""
def mutiple_birth(family_dict, individual_dict):
    for id, individual in individual_dict.items():
        if individual.famc != 'NA':
            siblings = family_dict[individual.famc].chil
            if len(siblings) > 5:
                birth = 0
                for child in siblings:
                    if individual_dict[child].birt.year != 'NA' and individual_dict[child].birt.month != 'NA' and individual_dict[child].birt.day != 'NA' and individual.birt.year != 'NA' and individual.birt.month != 'NA' and individual.birt.day != 'NA':
                        sib_age = date(int(float(individual_dict[child].birt.year)), int(float(individual_dict[child].birt.month)),
                                       int(float(individual_dict[child].birt.day)))
                        self_age = date(int(float(individual.birt.year)), int(float(individual.birt.month)),
                                        int(float(individual.birt.day)))
                        difference = abs(self_age - sib_age).days
                        if difference <= 1:
                            birth += 1
                        if birth > 5:
                            ErrorCollector.error_list.append(f"ERROR: US14: {id} has too many siblings born at the same time")
                            break

"""User story 18: sibling should not marry"""
def sibling_not_marry(family_dict, individual_dict):
    for id, individual in individual_dict.items():
        partner = 'NA'
        if individual.famc != 'NA' and individual.fams != 'NA':
            if individual.sex == 'M':
                partnerid = family_dict[individual.fams].wife
            else:
                partnerid = family_dict[individual.fams].husb
            if partnerid != 'NA':
                partner = individual_dict[partnerid]
                if individual.famc == partner.famc:
                    ErrorCollector.error_list.append(f"ERROR: US18: {id} should not marry with sibling {partnerid}")
        
"""User story 19: first cousin not marry"""
def first_cousin_not_marry(family_dict, individual_dict):
    def find_grandparents(family_dict, individual_dict, individual):
        gffID = 'NA'
        gfmID = 'NA'
        gmfID = 'NA'
        gmmID = 'NA'
        if individual.famc != 'NA':
            fatherID = family_dict[individual.famc].husb
            motherID = family_dict[individual.famc].wife
            if fatherID != 'NA':
                father = individual_dict[fatherID]
                if father.famc != 'NA':
                    gffID = family_dict[father.famc].husb
                    gfmID = family_dict[father.famc].wife
            if motherID != 'NA':
                mother = individual_dict[motherID]
                if mother.famc != 'NA':
                    gmfID = family_dict[mother.famc].husb
                    gmmID = family_dict[mother.famc].wife
        return gffID, gfmID, gmfID, gmmID
    
    for id, individual in individual_dict.items():
        partner = 'NA'
        if individual.famc != 'NA' and individual.fams != 'NA':
            if individual.sex == 'M':
                partnerid = family_dict[individual.fams].wife
            else:
                partnerid = family_dict[individual.fams].husb
            if partnerid != 'NA':
                partner = individual_dict[partnerid]
                indigp = find_grandparents(family_dict, individual_dict, individual)
                partgp = find_grandparents(family_dict, individual_dict, partner)
                for i in range(4):
                    if indigp[i] == partgp[i] and indigp[i] != 'NA':
                        ErrorCollector.error_list.append(f"ERROR: US19: {id} should not marry his cousin {partnerid}")
                        break

                                   
'''Sprint 4'''
'''User Story 20 : Aunts and Uncle'''


def Aunts_Uncles(family_dict, individual_dict):
    for id, individual in individual_dict.items():
        partner = 'NA'
        if individual.famc != 'NA' and individual.fams != 'NA':
            if individual.sex == 'M':
                partnerid = family_dict[individual.fams].wife
            else:
                partnerid = family_dict[individual.fams].husb
            if partnerid != 'NA':
                partner = individual_dict[partnerid]
                motherid = family_dict[individual.famc].wife
                if motherid != 'NA':
                    mother = individual_dict[motherid]
                    if partner.famc == mother.famc:
                        ErrorCollector.error_list.append(
                            f"ERROR: US20: {id} should not marry his aunt or uncle {partnerid}")
                fatherid = family_dict[individual.famc].husb
                if fatherid != 'NA':
                    father = individual_dict[fatherid]
                    if partner.famc == mother.famc:
                        ErrorCollector.error_list.append(
                            f"ERROR: US20: {id} should not marry his aunt or uncle {partnerid}")


'''User Story 21 : Correct gender for role'''


def Correct_gender(family_dict, individual_dict):
    for id, individual in individual_dict.items():
        if individual.fams != 'NA':
            if individual.sex == 'M':
                selfID = family_dict[individual.fams].husb
            else:
                selfID = family_dict[individual.fams].wife
            if individual != individual_dict[selfID]:
                ErrorCollector.error_list.append(f"ERROR: US21: {id} gender is not correct")

                                  

"""Shengda's Code Goes Here"""
'''Sprint 1'''
'''User Story 05: Marriage Before Death'''
def marriage_before_death(family_dict, individual_dict):
    US05_report = {}
    for fam, family in family_dict.items():
        husband_ID = family.husb
        wife_ID = family.wife
        marriage_date = family.marr.snake_year_month_day()
        if husband_ID != 'NA' and wife_ID != 'NA':
            husband_death_date = individual_dict[husband_ID].deat.snake_year_month_day()
            wife_death_date = individual_dict[wife_ID].deat.snake_year_month_day()
            if marriage_date != 'NA':
                if husband_death_date == 'NA' and wife_death_date == 'NA':
                    US05_report[fam] = True
                elif husband_death_date != 'NA' and wife_death_date == 'NA':
                    US05_report[fam] = husband_death_date >= marriage_date
                elif wife_death_date != 'NA' and husband_death_date == 'NA':
                    US05_report[fam] = wife_death_date >= marriage_date
                else:
                    first_death_date = husband_death_date if husband_death_date <= wife_death_date else wife_death_date
                    US05_report[fam] = first_death_date >= marriage_date
    for id, boolean in US05_report.items():
        if boolean == False:
            ErrorCollector.error_list.append(f"ERROR: FAMILY: US05: Family ID: {id} Marriage occur before death of either spouse.")
    #print('5',US05_report)
    return US05_report

'''User Story 07: Less Than 150 Years Old'''
def less_than_150_years_old(individual_dict):
    US07_report = {}
    for id, individual in individual_dict.items():
        indi_death_date = individual.deat.snake_year_month_day()
        indi_birth_date = individual.birt.snake_year_month_day()

        if indi_death_date != 'NA' and indi_birth_date != 'NA':
            today = date.today()
            year, month, day = indi_birth_date.split("-")
            year_dif = today.year - int(year) - ((today.month, today.day) < (int(month), int(day)))
            if year_dif > 150:
                US07_report[id] = False
            else:
                US07_report[id] = True
        else:
            if age_calculator(date.today(), individual.birt) != 'NA':
                if age_calculator(date.today(), individual.birt) > 150:
                    US07_report[id] = False
                else:
                    US07_report[id] = True
    for id, boolean in US07_report.items():
        if boolean == False:
            ErrorCollector.error_list.append(f"ERROR: INDIVIDUAL: US07: Individual ID: {id} "
                                             f"Death should be less than 150 years after birth for dead people, "
                                             f"and current date should be less than 150 years after birth for all living people")
    #print('7',US07_report)
    return US07_report

'''Sprint 2'''
'''User Story 10: Marriage after 14'''
def marriage_after_14(family_dict, individual_dict):
    US10_report = {}
    for fam, family in family_dict.items():
        #if individual.famc != 'NA':
        husband_ID = family.husb
        wife_ID = family.wife
        marriage_date = family.marr.snake_year_month_day()
        if marriage_date != 'NA':
            husband_birth_day = individual_dict[husband_ID].birt.snake_year_month_day()
            wife_birth_day = individual_dict[wife_ID].birt.snake_year_month_day()

            if calculate_year_dif(husband_birth_day, marriage_date) >= 14                and calculate_year_dif(wife_birth_day, marriage_date) >= 14:
                US10_report[fam] = True
            else:
                US10_report[fam] = [marriage_date, husband_birth_day, wife_birth_day]
    for id, boolean in US10_report.items():
        if boolean != True:
            ErrorCollector.error_list.append(f"ERROR: FAMILY: US10: Family ID: {id}, marriage date is {boolean[0]}, "
                  f"husband birth day is {boolean[1]}, wife birth day is {boolean[2]},"
                  f"married before 14 years old.")
    #print('10', US10_report)
    return US10_report

#User Story 10 helper
def calculate_year_dif(one_date_string, another_date_string):
    if one_date_string != 'NA' and another_date_string != 'NA':
        year1, month1, day1 = one_date_string.split("-")
        year2, month2, day2 = another_date_string.split("-")
        year_dif = int(year2) - int(year1) - ((int(month2), int(day2)) < (int(month1), int(day1)))
        return year_dif
    return True

'''User Story 31: List living single'''
def list_living_single(family_dict, individual_dict):
    US31_report = {}

    for indi, individual in individual_dict.items():
        indi_birth_date = individual.birt.snake_year_month_day()
        indi_death_date = individual.deat.snake_year_month_day()
        indi_spouse = individual.fams

        if indi_birth_date != 'NA' and indi_death_date == 'NA':
            if age_calculator(date.today(), individual.birt) >= 30 and indi_spouse == 'NA':
                US31_report[indi] = [age_calculator(date.today(), individual.birt), indi_spouse]
            else:
                US31_report[indi] = False
    for id, boolean in US31_report.items():
        if boolean != False:
            ErrorCollector.error_list.append(f"ERROR: INDIVIDUAL: US31: Individual ID: {id}, Age is {boolean[0]}, FamilyID is {boolean[1]}, "
                  f"who is living people over 30 but have never been married")
    #print('31',US31_report)
    return US31_report

'''Sprint 3'''
"""User Story 35: List all people in a GEDCOM file who were born in the last 30 days"""
def List_recent_births(individual_dict):
    US35_report = {}

    for indi, individual in individual_dict.items():
        indi_birth_date = individual.birt.snake_year_month_day()
        if indi_birth_date != 'NA':
            string_date_list = indi_birth_date.split('-')
            year = int(string_date_list[0])
            month = int(string_date_list[1])
            day = int(string_date_list[2])
            current_time = datetime.datetime.now()
            birth_date = datetime.datetime(year, month, day)
            difference_days = (current_time - birth_date).days
            if difference_days <= 30:
                US35_report.setdefault(indi, []).extend([birth_date, current_time])

    for id, date in US35_report.items():
        ErrorCollector.error_list.append(f"ERROR: INDIVIDUAL: US35: Individual ID: {id}, birth date is {date[0]}, "
            f"current time is {date[1]}, is the people who were born in the last 30 days.")
    #print('35', US35_report)
    return US35_report

"""User Story 36: List all people in a GEDCOM file who died in the last 30 days"""
def List_recent_deaths(individual_dict):
    US36_report = {}

    for indi, individual in individual_dict.items():
        indi_death_date = individual.deat.snake_year_month_day()
        if indi_death_date != 'NA':
            string_date_list = indi_death_date.split('-')
            year = int(string_date_list[0])
            month = int(string_date_list[1])
            day = int(string_date_list[2])
            current_time = datetime.datetime.now()
            death_date = datetime.datetime(year, month, day)
            difference_days = (current_time - death_date).days
            if difference_days <= 30:
                US36_report.setdefault(indi, []).extend([death_date, current_time])

    for id, date in US36_report.items():
        ErrorCollector.error_list.append(f"ERROR: INDIVIDUAL: US36: Individual ID: {id}, death date is {date[0]}, "
            f"current time is {date[1]}, is the people who were died in the last 30 days.")
    #print('36', US36_report)
    return US36_report

'''Sprint 4'''
"""US 37: List all living spouses and descendants of people 
in a GEDCOM file who died in the last 30 days"""
def List_recent_survivors(family_dict, individual_dict):
    US37_report = {}
    for fam, family in family_dict.items():
        husband_ID = family.husb
        wife_ID = family.wife
        children = family.chil
        if husband_ID != 'NA' and wife_ID != 'NA':
            husband_death_date = individual_dict[husband_ID].deat.snake_year_month_day()
            wife_death_date = individual_dict[wife_ID].deat.snake_year_month_day()
            if husband_death_date != 'NA' and wife_death_date == 'NA':
                deat_datetime = datetime.datetime.strptime(husband_death_date,"%Y-%m-%d")
                current_datetime = datetime.datetime.now()
                dif_datetime = current_datetime - deat_datetime
                if dif_datetime.days <= 30:
                    US37_report.setdefault(fam, []).append([husband_death_date, wife_death_date, children])
                    ErrorCollector.error_list.append(f"ERROR: FAMILY: US37: husband ID: {husband_ID}, death date is {husband_death_date}, "
            f"the spouse is {wife_ID}, {children} are the descendants of husband in a GEDCOM file who died in the last 30 days")

        if husband_ID != 'NA' and wife_ID != 'NA':
            husband_death_date = individual_dict[husband_ID].deat.snake_year_month_day()
            wife_death_date = individual_dict[wife_ID].deat.snake_year_month_day()
            if wife_death_date != 'NA' and husband_death_date == 'NA':
                deat_datetime = datetime.datetime.strptime(wife_death_date,"%Y-%m-%d")
                current_datetime = datetime.datetime.now()
                dif_datetime = current_datetime - deat_datetime
                if dif_datetime.days <= 30:
                    US37_report.setdefault(fam, []).append([husband_death_date, wife_death_date, children])
                    ErrorCollector.error_list.append(f"ERROR: FAMILY: US37: wife ID: {wife_ID}, death date is {wife_death_date}, "
            f"the spouse is {husband_ID}, {children} are the descendants of husband in a GEDCOM file who died in the last 30 days")
    #print("37", US37_report)
    return US37_report

"""US38: List upcoming birthdays: 
List all living people in a GEDCOM file whose birthdays occur in the next 30 days"""
def List_upcoming_birthdays(individual_dict):
    US38_report = {}
    for indi, individual in individual_dict.items():
        indi_birth_date = individual.birt.snake_year_month_day()
        if indi_birth_date != 'NA':
            indi_birth_datetime = datetime.datetime.strptime(indi_birth_date, "%Y-%m-%d")
            current_datetime = datetime.datetime.now()
            year_dif = current_datetime.year - indi_birth_datetime.year
            month_dif = current_datetime.month - indi_birth_datetime.month
            days_dif = current_datetime.day - indi_birth_datetime.day
            if year_dif >= 1 and month_dif == 0 and days_dif <= 30:
                US38_report.setdefault(indi, []).append([indi_birth_datetime, current_datetime])
                ErrorCollector.error_list.append(
                    f"ERROR: INDIVIDUAL: US38: individual id is: {indi}, birth date is {indi_birth_datetime}, "
                    f"current time is {current_datetime}, is the living people in a GEDCOM file whose birthdays occur in the next 30 days")
    #print("38", US38_report)
    return US38_report

"""Xiangyu's Code Goes Here"""
'''Sprint 1'''
'''User Story 15: Fewer Than 150 Siblings'''
def fewer_than_15_siblings(family_dict):
    US15_report = {}
    for id, family in family_dict.items():
        if len(family.chil) >= 15:
            children = list(family.chil)
            US15_report.setdefault(id, []).extend(children)
            ErrorCollector.error_list.append(f"ERROR: US15: Family {id} has {children} children more than 15.")
    #print('15', US15_report)
    return US15_report


'''User Story 24: Unique Families By Spouses'''
def unique_families_by_spouses(family_dict, individual_dict):
    US24_report = {}
    for fam, family in family_dict.items():
        husband_id = family.husb
        wife_id = family.wife
        if wife_id != 'NA' and husband_id != 'NA':
            husband_name = individual_dict[husband_id].name
            wife_name = individual_dict[wife_id].name
            marriage_date = family.marr.snake_year_month_day()

            if marriage_date != 'NA':
                if marriage_date in US24_report.keys():
                    if US24_report[marriage_date][0] == husband_name\
                            or US24_report[marriage_date][1] == wife_name:
                        ErrorCollector.error_list.append(f"ERROR: US24: family id is {fam}, husband name  is {husband_name}, "
                                                         f"wife name is {wife_name}, marriage date is {marriage_date}, is not unique Families By Spouses")
                else:
                    US24_report[marriage_date] = [husband_name, wife_name]
    #print('24', US24_report)
    return US24_report

'''Sprint 2'''
'''User Story 26: Corresponding entries'''
def corresponding_entries(family_dict, individual_dict):
    US26_report ={}
    for id, family in family_dict.items():
        husband_id = family.husb
        wife_id = family.wife
        children = family.chil
        if wife_id != 'NA' and husband_id != 'NA':
            if individual_dict[husband_id].fams != id:
                ErrorCollector.error_list.append(f"ERROR: US26: family id is {id}, wife is {wife_id}, husband is {husband_id} in family record, but in individual record, individual {husband_id} is in {individual_dict[husband_id].fams} family")
                US26_report.setdefault(id, []).append(individual_dict[husband_id].fams)
            if individual_dict[wife_id].fams != id:
                ErrorCollector.error_list.append(f"ERROR: US26: family id is {id}, wife is {wife_id}, husband is {husband_id} in family record, but in individual record, individual {wife_id} is in {individual_dict[husband_id].fams} family")
                US26_report.setdefault(id, []).append(individual_dict[wife_id].fams)
            if children != 'NA':
                for child in children:
                    if child in individual_dict:
                        if child != 'NA':
                            if individual_dict[child].famc != id:
                                ErrorCollector.error_list.append(f"ERROR: US26: family id is {id}, child {child} in family record, but in individual record, child {child} is in {individual_dict[child].famc} family")
                        US26_report.setdefault(id, []).append(individual_dict[child].famc)

    #print("us26", US26_report)
    return US26_report

'''User Story 32: List multiple births'''
def list_multiple_births(individual_dict):
    US32_report = {}
    for id, individual in individual_dict.items():
        birth_date = individual.birt.snake_year_month_day()
        US32_report.setdefault(birth_date, []).append(id)
    for birth_date in US32_report:
        if birth_date != 'NA':
            if len(US32_report[birth_date]) > 1:
                ErrorCollector.error_list.append(f"ERROR: INDIVIDUAL: US32: birth day {birth_date} has multiple births {US32_report[birth_date]}")

    #print("us32", US32_report)
    return US32_report

'''Sprint 3'''
'''User Story 33: List all orphaned children (both parents dead and child < 18 years old) in a GEDCOM file'''
def list_orphans(family_dict, individual_dict):
    US33_report = {}
    for fam, family in family_dict.items():
        husband_ID = family.husb
        wife_ID = family.wife
        children_set = family.chil
        if husband_ID != 'NA' and wife_ID != 'NA':
            husband_death_day = individual_dict[husband_ID].deat.snake_year_month_day()
            wife_death_day = individual_dict[wife_ID].deat.snake_year_month_day()
            if husband_death_day != 'NA' and wife_death_day != 'NA':
                for child_id in children_set:
                    age = age_calculator(datetime.date.today(), individual_dict[child_id].birt)
                    if age != 'NA':
                        if age < 18:
                            US33_report.setdefault(child_id, []).extend([husband_death_day, wife_death_day, age])

    for id, date in US33_report.items():
        ErrorCollector.error_list.append(f"ERROR: INDIVIDUAL: US33: Individual ID: {id} is orphan, husband death date is {date[0]}, "
            f"wife death day is {date[1]}, child's age is {date[2]},"
                  f"married before 14 years old.")
    #print('33', US33_report)
    return US33_report

'''User Story 34: List all couples who were married when the older spouse was more than twice as old as the younger spouse'''
def list_large_age_differences(family_dict, individual_dict):
    US34_report = {}
    for fam, family in family_dict.items():
        husband_ID = family.husb
        wife_ID = family.wife
        if husband_ID != 'NA' and wife_ID != 'NA':
            husband_age = age_calculator(datetime.date.today(),individual_dict[husband_ID].birt)
            wife_age = age_calculator(datetime.date.today(),individual_dict[wife_ID].birt)
            if husband_age != 'NA' and wife_age != 'NA':
                if husband_age >= 2 * wife_age or wife_age >= 2 * husband_age:
                    US34_report.setdefault(fam, []).extend([husband_age, wife_age])

    for id, date in US34_report.items():
        ErrorCollector.error_list.append(f"ERROR: FAMILY: US34: Family ID: {id}, husband's age is {date[0]}, "
            f"wife's age is {date[1]}, are couples who were married the older spouse was more than twice as old as the younger spouse.")
    #print('34', US34_report)
    return US34_report

'''Sprint 4'''
"""US39: List all living couples 
in a GEDCOM file whose marriage anniversaries occur in the next 30 days"""
def List_upcoming_anniversaries(family_dict):
    US39_report = {}

    for fam, family in family_dict.items():
        husband_ID = family.husb
        wife_ID = family.wife
        if husband_ID != 'NA' and wife_ID != 'NA':
            marriage_date = family.marr.snake_year_month_day()
            if marriage_date != 'NA':
                marriage_datetime = datetime.datetime.strptime(marriage_date, "%Y-%m-%d")
                current_datetime = datetime.datetime.now()
                year_dif = current_datetime.year - marriage_datetime.year
                month_dif = current_datetime.month - marriage_datetime.month
                days_dif = current_datetime.day - marriage_datetime.day
                if year_dif >= 1 and month_dif == 0 and days_dif <= 30:
                    US39_report.setdefault(fam, []).append([husband_ID, wife_ID])
                    ErrorCollector.error_list.append(f"ERROR: FAMILY: US39: Family ID: {fam}, husband's ID is {husband_ID}, "
                    f"wife's ID is {wife_ID}, whose marriage anniversaries occur in the next 30 days.")
    #print("39", US39_report)
    return US39_report

"""US40: Include input line numbers 
List line numbers from GEDCOM source file when reporting errors"""
def Include_input_line_numbers(list):
    US40_report = {}
    for error_string in list:
        string_list = error_string.split(":")
        for us_id in string_list:
            real_id = us_id.strip().upper()
            if real_id.startswith("US"):
                id = real_id[2:]
                US40_report[id] = US40_report.get(id, 0) + 1
    for id, cnt in US40_report.items():
        ErrorCollector.error_list.append(f"US40: User Story ID is {id}, the numbers of the errors are {cnt}")
    #print("40", US40_report)
    return US40_report
"""Main Function"""
def main():
    '''Operations in Order'''
    # 1. Read the file
    # 2. Filter the file
    # 3. Das Dictionary Für Individual Objects & Das Dictionary Für Family Objects
    # 4. Draw Pretty Tables
    # 5. Print all the errors
    # 6. Export for testing
    #
    file_path = '555Project(updates-often).ged'
    #
    '''Dictionaries of Individuals and Families'''
    individuals = {}
    families = {}
    # 1. Read the file
    with open(file_path, 'r') as file:
        unfiltered_file = file.readlines()
        # 2. Filter the file
        filter_file(unfiltered_file, individuals, families)

    '''Uncomment these if you want to see the individuals and families in raw format'''
    # for key, value in individuals.items():
    #     print(key, value)
    # for key, value in families.items():
    #     print(key, value)

    # 3. Das Dictionary Für Individual Objects & Das Dictionary Für Family Objects
    individual_dict = {}
    family_dict = {}
    '''the next 2 lines fill in the previous 2 lines individual_dict and family_dict'''
    raw_individuals_to_structured_dict(individuals, individual_dict)
    raw_families_to_structured_dict(families, family_dict)

    """User Stories Goes Here"""
    # manual GED creation: US17, US21, US22
    '''Hanqing Sprint 1: US01, US02'''
    dates_before_current_date(individual_dict, family_dict) # US01
    birth_before_marriage(individual_dict, family_dict) # US02
    '''Hanqing Sprint 2: US03, US08'''
    birth_before_death(individual_dict) # US03
    birth_before_marriage_of_parents(individual_dict, family_dict) # US08
    '''Hanqing Sprint 3: US09, US16'''
    birth_before_parents_death(individual_dict, family_dict) # US09
    male_last_names(individual_dict, family_dict) #US16
    '''Hanqing Sprint 4: US23, US28'''
    unique_name_and_birth_date(individual_dict) # US23
    order_siblings_by_age(individual_dict, family_dict) # US28

    '''Jigar Sprint 1: US04, US06'''
    marriage_before_divorce(family_dict) # US04
    divorce_before_death(family_dict, individual_dict) # US06
    '''Jigar Sprint 2: US17, US22'''
    no_marriage_to_children(family_dict, individual_dict) # US17
    unique_ids(unfiltered_file) #US22
    '''Jigar Sprint 3: US25, US27'''
    unique_first_name(family_dict, individual_dict) #US25
    include_individual_ages(individual_dict) #US27
    '''Jigar Sprint 4: US29, US30'''
    list_deceased(individual_dict)  #US29
    list_living_married(family_dict, individual_dict)  #US30                               

    '''Shengda Sprint 1: US05, US07'''
    marriage_before_death(family_dict, individual_dict) # US05
    less_than_150_years_old(individual_dict) # US07
    '''Shengda Sprint 2: US10, US31'''
    marriage_after_14(family_dict, individual_dict) # US10
    list_living_single(family_dict, individual_dict) # US31
    '''shengda Sprint 3: US35, US36'''
    List_recent_births(individual_dict)
    List_recent_deaths(individual_dict)
    '''shengda Sprint 4: US37, US38'''
    List_recent_survivors(family_dict, individual_dict)
    List_upcoming_birthdays(individual_dict)

    '''Haoran Sprint 1: US11, US12'''
    no_bigamy(family_dict, individual_dict) # US11
    parents_not_too_old(family_dict, individual_dict) # US12
    '''Haoran Sprint 2: US13, US14'''
    siblings_spacing(family_dict, individual_dict) # US13
    mutiple_birth(family_dict, individual_dict) # US14
    '''Haoran Sprint 3: US18, US19'''
    sibling_not_marry(family_dict, individual_dict) # US18
    first_cousin_not_marry(family_dict, individual_dict) # US19
    '''Haoran Sprint 4: US20, US21'''
    Aunts_Uncles(family_dict, individual_dict)  #20
    Correct_gender(family_dict, individual_dict)  #21                               

    '''Xiangyu Sprint 1: US15, US24'''
    fewer_than_15_siblings(family_dict) # US15
    unique_families_by_spouses(family_dict, individual_dict) # US24
    '''Xiangyu Sprint 2: US26, US32'''
    corresponding_entries(family_dict, individual_dict) # US26
    list_multiple_births(individual_dict) # US32
    '''Xiangyu Sprint 3: US33, US34'''
    list_orphans(family_dict, individual_dict)
    list_large_age_differences(family_dict, individual_dict)
    '''Xiangyu Sprint 4: US39, US40'''
    List_upcoming_anniversaries(family_dict)
    Include_input_line_numbers(ErrorCollector.error_list)


    '''Uncomment these if you want to see the individual and family objects created from Individual and Family class'''
    # for key, value in individual_dict.items():
    #     print(key, value.name, value.sex, value.birt.snake_year_month_day(), value.deat.snake_year_month_day(), value.famc, value.fams)
    # for key, value in family_dict.items():
    #     print(key, value.marr.snake_year_month_day(), value.husb, value.wife, value.chil, value.div.snake_year_month_day())

    # 4. Draw Pretty Tables
    draw_individual_prettytable(individual_dict)
    draw_family_prettytable(family_dict, individual_dict)

    # 5. Print all errors
    for error in ErrorCollector.error_list:
        print(error)

    # 6. Export for testing
    return ErrorCollector


"""Run Main Function"""
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





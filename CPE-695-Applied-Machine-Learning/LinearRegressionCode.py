import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# """Tune Parameters"""
# df_all = pd.read_csv('NYPD_Complaint_Data_Historic2010.csv')
# df_all = df_all[['RPT_DT', 'LAW_CAT_CD']]
# df_all = df_all.dropna()

# df_all['RPT_DT'] = pd.to_datetime(df_all['RPT_DT'])
# df_all['RPT_DT'] = df_all['RPT_DT'].map(dt.datetime.toordinal)

# x_values = sorted(df_all['RPT_DT'].unique())

# # Setting up dictionaries
# violation_date_count_dict = {}
# misdemeanor_date_count_dict = {}
# felony_date_count_dict = {}

# for value in x_values:
#     violation_date_count_dict[value] = 0
#     misdemeanor_date_count_dict[value] = 0
#     felony_date_count_dict[value] = 0

# for index, row in df_all.iterrows():
#     if row['LAW_CAT_CD'] == 'VIOLATION':
#         violation_date_count_dict[row['RPT_DT']] += 1
#     elif row['LAW_CAT_CD'] == 'MISDEMEANOR':
#         misdemeanor_date_count_dict[row['RPT_DT']] += 1
#     elif row['LAW_CAT_CD'] == 'FELONY':
#         felony_date_count_dict[row['RPT_DT']] += 1

# # x_values = map(dt.datetime.fromordinal, x_values)

# violation_ys = []
# misdemeanor_ys = []
# felony_ys = []
# for key, value in violation_date_count_dict.items():
#     violation_ys.append(value)
# for key, value in misdemeanor_date_count_dict.items():
#     misdemeanor_ys.append(value)
# for key, value in felony_date_count_dict.items():
#     felony_ys.append(value)

# independentVariables = pd.DataFrame(x_values, columns =['Date'])
# dependentVariables_v = pd.DataFrame(violation_ys, columns=['Violations'])
# dependentVariables_m = pd.DataFrame(misdemeanor_ys, columns=['Misdemeanors'])
# dependentVariables_f = pd.DataFrame(felony_ys, columns=['Felonies'])


# X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(independentVariables, dependentVariables_v, test_size=0.2)
# X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(independentVariables, dependentVariables_m, test_size=0.2)
# X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(independentVariables, dependentVariables_f, test_size=0.2)

# # accuracy = accuracy_score(y_test, y_test_pred)
# # print(accuracy)

# m = [1,2,3,4,5]
# MSE_Vio = []
# MSE_Mis = []
# MSE_Fel = []

# MSE_Vio_train = []
# MSE_Mis_train = []
# MSE_Fel_train = []

# # for val in m:
# #     X_v_train = X_train_v.values.flatten()
# #     y_v_train = y_train_v.values.flatten()
# #     z_v_train = np.polyfit(X_v_train, y_v_train, val)
# #     p_v_train = np.poly1d(z_v_train)
# #     MSE_value_train = np.square(np.subtract(y_v_train, p_v_train(X_v_train))).mean()
# #     MSE_Vio_train.append(MSE_value_train)

# #     X_v = X_test_v.values.flatten()
# #     y_v = y_test_v.values.flatten()
# #     # z_v = np.polyfit(X_v, y_v, val)
# #     # p_v = np.poly1d(z_v)
# #     MSE_value = np.square(np.subtract(y_v, p_v_train(X_v))).mean()
# #     MSE_Vio.append(MSE_value)


# # plt.plot(m, MSE_Vio, '-', label="Testing")
# # plt.plot(m, MSE_Vio_train, '-', label="Training")
# # plt.title("MSE Vs. Order M for Violation")
# # plt.xlabel("m")
# # plt.ylabel("MSE")
# # plt.legend(loc="upper right")
# # plt.show()

# # for val in m:
# #     X_m_train = X_train_m.values.flatten()
# #     y_m_train = y_train_m.values.flatten()
# #     z_m_train = np.polyfit(X_m_train, y_m_train, val)
# #     p_m_train = np.poly1d(z_m_train)
# #     MSE_value_train = np.square(np.subtract(y_m_train, p_m_train(X_m_train))).mean()
# #     MSE_Mis_train.append(MSE_value_train)

# #     X_m = X_test_m.values.flatten()
# #     y_m = y_test_m.values.flatten()
# #     # z_m = np.polyfit(X_m, y_m, val)
# #     # p_m = np.poly1d(z_m)
# #     MSE_value = np.square(np.subtract(y_m, p_m_train(X_m))).mean()
# #     MSE_Mis.append(MSE_value)

# #     # X_m_train = X_train_m.values.flatten()
# #     # y_m_train = y_train_m.values.flatten()
# #     # z_m_train = np.polyfit(X_m_train, y_m_train, val)
# #     # p_m_train = np.poly1d(z_m_train)
# #     # MSE_value_train = np.square(np.subtract(y_m_train, p_m_train(X_m_train))).mean()
# #     # MSE_Mis_train.append(MSE_value_train)

# # plt.plot(m, MSE_Mis, '-', label="Testing")
# # plt.plot(m, MSE_Mis_train, '-', label="Training")
# # plt.title("MSE Vs. Order M for Misdemeanor")
# # plt.xlabel("m")
# # plt.ylabel("MSE")
# # plt.legend(loc="upper right")
# # plt.show()

# for val in m:
#     X_f_train = X_train_f.values.flatten()
#     y_f_train = y_train_f.values.flatten()
#     z_f_train = np.polyfit(X_f_train, y_f_train, val)
#     p_f_train = np.poly1d(z_f_train)
#     MSE_value_train = np.square(np.subtract(y_f_train, p_f_train(X_f_train))).mean()
#     MSE_Fel_train.append(MSE_value_train)

#     X_f = X_test_f.values.flatten()
#     y_f = y_test_f.values.flatten()
#     # z_m = np.polyfit(X_m, y_m, val)
#     # p_m = np.poly1d(z_m)
#     MSE_value = np.square(np.subtract(y_f, p_f_train(X_f))).mean()
#     MSE_Fel.append(MSE_value)

#     # X_m_train = X_train_m.values.flatten()
#     # y_m_train = y_train_m.values.flatten()
#     # z_m_train = np.polyfit(X_m_train, y_m_train, val)
#     # p_m_train = np.poly1d(z_m_train)
#     # MSE_value_train = np.square(np.subtract(y_m_train, p_m_train(X_m_train))).mean()
#     # MSE_Mis_train.append(MSE_value_train)

# plt.plot(m, MSE_Fel, '-', label="Testing")
# plt.plot(m, MSE_Fel_train, '-', label="Training")
# plt.title("MSE Vs. Order M for Felony")
# plt.xlabel("m")
# plt.ylabel("MSE")
# plt.legend(loc="upper right")
# plt.show()


# # for val in m:
# #     X_m = X_test_m.values.flatten()
# #     y_m = y_test_m.values.flatten()
# #     z_m = np.polyfit(X_m, y_m, val)
# #     p_m = np.poly1d(z_m)
# #     MSE_value = np.square(np.subtract(y_m, p_m(X_m))).mean()
# #     MSE_Mis.append(MSE_value)

# # plt.plot(m, MSE_Mis, 'ro', m, MSE_Mis, '-')
# # plt.title("MSE Vs. Order M for Misdemeanor")
# # plt.xlabel("m")
# # plt.ylabel("MSE")
# # plt.show()

# for val in m:
#     X_f = X_test_f.values.flatten()
#     y_f = y_test_f.values.flatten()

global_m_value = 3


"""Bronx"""
df_bronx = pd.read_csv('/Users/hanqingliu/Downloads/NYPD_Complaint_Data_Historic-Bronx.csv', low_memory=False)
df_bronx = df_bronx[['RPT_DT', 'LAW_CAT_CD']]
df_bronx = df_bronx.dropna()

df_bronx['RPT_DT'] = pd.to_datetime(df_bronx['RPT_DT'])
df_bronx['RPT_DT'] = df_bronx['RPT_DT'].map(dt.datetime.toordinal)

x_values = sorted(df_bronx['RPT_DT'].unique())

# Setting up dictionaries
violation_date_count_dict = {}
misdemeanor_date_count_dict = {}
felony_date_count_dict = {}

for value in x_values:
    violation_date_count_dict[value] = 0
    misdemeanor_date_count_dict[value] = 0
    felony_date_count_dict[value] = 0

for index, row in df_bronx.iterrows():
    if row['LAW_CAT_CD'] == 'VIOLATION':
        violation_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'MISDEMEANOR':
        misdemeanor_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'FELONY':
        felony_date_count_dict[row['RPT_DT']] += 1

# x_values = map(dt.datetime.fromordinal, x_values)

violation_ys = []
misdemeanor_ys = []
felony_ys = []
for key, value in violation_date_count_dict.items():
    violation_ys.append(value)
for key, value in misdemeanor_date_count_dict.items():
    misdemeanor_ys.append(value)
for key, value in felony_date_count_dict.items():
    felony_ys.append(value)

independentVariables = pd.DataFrame(x_values, columns=['Date'])
dependentVariables_v = pd.DataFrame(violation_ys, columns=['Violations'])
dependentVariables_m = pd.DataFrame(misdemeanor_ys, columns=['Misdemeanors'])
dependentVariables_f = pd.DataFrame(felony_ys, columns=['Felonies'])

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(independentVariables, dependentVariables_v, test_size=0.2)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(independentVariables, dependentVariables_m, test_size=0.2)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(independentVariables, dependentVariables_f, test_size=0.2)


x_axis_v = X_test_v['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_m = X_test_m['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_f = X_test_f['Date'].map(dt.datetime.fromordinal).values.flatten()

plt.title(label="Bronx")
X_v = X_test_v.values.flatten()
y_v = y_test_v.values.flatten()
z_v = np.polyfit(X_v, y_v, global_m_value)
p_v = np.poly1d(z_v)
plt.plot(x_axis_v, p_v(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten()
y_m = y_test_m.values.flatten()
z_m = np.polyfit(X_m, y_m, global_m_value)
p_m = np.poly1d(z_m)
plt.plot(x_axis_m, p_m(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten()
y_f = y_test_f.values.flatten()
z_f = np.polyfit(X_f, y_f, global_m_value)
p_f = np.poly1d(z_f)
plt.plot(x_axis_f, p_f(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Manhattan"""
df_manhattan = pd.read_csv('/Users/hanqingliu/Downloads/NYPD_Complaint_Data_Historic-Manhattan.csv', low_memory=False)
df_manhattan = df_manhattan[['RPT_DT', 'LAW_CAT_CD']]
df_manhattan = df_manhattan.dropna()

df_manhattan['RPT_DT'] = pd.to_datetime(df_manhattan['RPT_DT'])
df_manhattan['RPT_DT'] = df_manhattan['RPT_DT'].map(dt.datetime.toordinal)

x_values = sorted(df_manhattan['RPT_DT'].unique())

# Setting up dictionaries
violation_date_count_dict = {}
misdemeanor_date_count_dict = {}
felony_date_count_dict = {}

for value in x_values:
    violation_date_count_dict[value] = 0
    misdemeanor_date_count_dict[value] = 0
    felony_date_count_dict[value] = 0

for index, row in df_manhattan.iterrows():
    if row['LAW_CAT_CD'] == 'VIOLATION':
        violation_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'MISDEMEANOR':
        misdemeanor_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'FELONY':
        felony_date_count_dict[row['RPT_DT']] += 1

violation_ys = []
misdemeanor_ys = []
felony_ys = []
for key, value in violation_date_count_dict.items():
    violation_ys.append(value)
for key, value in misdemeanor_date_count_dict.items():
    misdemeanor_ys.append(value)
for key, value in felony_date_count_dict.items():
    felony_ys.append(value)

independentVariables = pd.DataFrame(x_values, columns =['Date'])
dependentVariables_v = pd.DataFrame(violation_ys, columns=['Violations'])
dependentVariables_m = pd.DataFrame(misdemeanor_ys, columns=['Misdemeanors'])
dependentVariables_f = pd.DataFrame(felony_ys, columns=['Felonies'])

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(independentVariables, dependentVariables_v, test_size=0.2)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(independentVariables, dependentVariables_m, test_size=0.2)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(independentVariables, dependentVariables_f, test_size=0.2)

x_axis_v = X_test_v['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_m = X_test_m['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_f = X_test_f['Date'].map(dt.datetime.fromordinal).values.flatten()

plt.title(label="Manhattan")
X_v = X_test_v.values.flatten()
y_v = y_test_v.values.flatten()
z_v = np.polyfit(X_v, y_v, global_m_value)
p_v = np.poly1d(z_v)
plt.plot(x_axis_v, p_v(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten()
y_m = y_test_m.values.flatten()
z_m = np.polyfit(X_m, y_m, global_m_value)
p_m = np.poly1d(z_m)
plt.plot(x_axis_m, p_m(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten()
y_f = y_test_f.values.flatten()
z_f = np.polyfit(X_f, y_f, global_m_value)
p_f = np.poly1d(z_f)
plt.plot(x_axis_f, p_f(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Staten Island"""
df_staten_island = pd.read_csv('/Users/hanqingliu/Downloads/NYPD_Complaint_Data_Historic-Staten_Island.csv', low_memory=False)
df_staten_island = df_staten_island[['RPT_DT', 'LAW_CAT_CD']]
df_staten_island = df_staten_island.dropna()

df_staten_island['RPT_DT'] = pd.to_datetime(df_staten_island['RPT_DT'])
df_staten_island['RPT_DT'] = df_staten_island['RPT_DT'].map(dt.datetime.toordinal)

x_values = sorted(df_staten_island['RPT_DT'].unique())

# Setting up dictionaries
violation_date_count_dict = {}
misdemeanor_date_count_dict = {}
felony_date_count_dict = {}

for value in x_values:
    violation_date_count_dict[value] = 0
    misdemeanor_date_count_dict[value] = 0
    felony_date_count_dict[value] = 0

for index, row in df_staten_island.iterrows():
    if row['LAW_CAT_CD'] == 'VIOLATION':
        violation_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'MISDEMEANOR':
        misdemeanor_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'FELONY':
        felony_date_count_dict[row['RPT_DT']] += 1

violation_ys = []
misdemeanor_ys = []
felony_ys = []
for key, value in violation_date_count_dict.items():
    violation_ys.append(value)
for key, value in misdemeanor_date_count_dict.items():
    misdemeanor_ys.append(value)
for key, value in felony_date_count_dict.items():
    felony_ys.append(value)

independentVariables = pd.DataFrame(x_values, columns =['Date'])
dependentVariables_v = pd.DataFrame(violation_ys, columns=['Violations'])
dependentVariables_m = pd.DataFrame(misdemeanor_ys, columns=['Misdemeanors'])
dependentVariables_f = pd.DataFrame(felony_ys, columns=['Felonies'])

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(independentVariables, dependentVariables_v, test_size=0.2)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(independentVariables, dependentVariables_m, test_size=0.2)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(independentVariables, dependentVariables_f, test_size=0.2)

x_axis_v = X_test_v['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_m = X_test_m['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_f = X_test_f['Date'].map(dt.datetime.fromordinal).values.flatten()

plt.title(label="Staten Island")
X_v = X_test_v.values.flatten()
y_v = y_test_v.values.flatten()
z_v = np.polyfit(X_v, y_v, global_m_value)
p_v = np.poly1d(z_v)
plt.plot(x_axis_v, p_v(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten()
y_m = y_test_m.values.flatten()
z_m = np.polyfit(X_m, y_m, global_m_value)
p_m = np.poly1d(z_m)
plt.plot(x_axis_m, p_m(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten()
y_f = y_test_f.values.flatten()
z_f = np.polyfit(X_f, y_f, global_m_value)
p_f = np.poly1d(z_f)
plt.plot(x_axis_f, p_f(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Brooklyn"""
df_brooklyn = pd.read_csv('/Users/hanqingliu/Downloads/NYPD_Complaint_Data_Historic-Brooklyn.csv', low_memory=False)
df_brooklyn = df_brooklyn[['RPT_DT', 'LAW_CAT_CD']]
df_brooklyn = df_brooklyn.dropna()

df_brooklyn['RPT_DT'] = pd.to_datetime(df_brooklyn['RPT_DT'])
df_brooklyn['RPT_DT'] = df_brooklyn['RPT_DT'].map(dt.datetime.toordinal)

x_values = sorted(df_brooklyn['RPT_DT'].unique())

# Setting up dictionaries
violation_date_count_dict = {}
misdemeanor_date_count_dict = {}
felony_date_count_dict = {}

for value in x_values:
    violation_date_count_dict[value] = 0
    misdemeanor_date_count_dict[value] = 0
    felony_date_count_dict[value] = 0

for index, row in df_brooklyn.iterrows():
    if row['LAW_CAT_CD'] == 'VIOLATION':
        violation_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'MISDEMEANOR':
        misdemeanor_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'FELONY':
        felony_date_count_dict[row['RPT_DT']] += 1

violation_ys = []
misdemeanor_ys = []
felony_ys = []
for key, value in violation_date_count_dict.items():
    violation_ys.append(value)
for key, value in misdemeanor_date_count_dict.items():
    misdemeanor_ys.append(value)
for key, value in felony_date_count_dict.items():
    felony_ys.append(value)

independentVariables = pd.DataFrame(x_values, columns =['Date'])
dependentVariables_v = pd.DataFrame(violation_ys, columns=['Violations'])
dependentVariables_m = pd.DataFrame(misdemeanor_ys, columns=['Misdemeanors'])
dependentVariables_f = pd.DataFrame(felony_ys, columns=['Felonies'])

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(independentVariables, dependentVariables_v, test_size=0.2)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(independentVariables, dependentVariables_m, test_size=0.2)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(independentVariables, dependentVariables_f, test_size=0.2)

x_axis_v = X_test_v['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_m = X_test_m['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_f = X_test_f['Date'].map(dt.datetime.fromordinal).values.flatten()

plt.title(label="Brooklyn")
X_v = X_test_v.values.flatten()
y_v = y_test_v.values.flatten()
z_v = np.polyfit(X_v, y_v, global_m_value)
p_v = np.poly1d(z_v)
plt.plot(x_axis_v, p_v(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten()
y_m = y_test_m.values.flatten()
z_m = np.polyfit(X_m, y_m, global_m_value)
p_m = np.poly1d(z_m)
plt.plot(x_axis_m, p_m(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten()
y_f = y_test_f.values.flatten()
z_f = np.polyfit(X_f, y_f, global_m_value)
p_f = np.poly1d(z_f)
plt.plot(x_axis_f, p_f(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Queens"""
df_queens = pd.read_csv('/Users/hanqingliu/Downloads/NYPD_Complaint_Data_Historic-Queens.csv', low_memory=False)
df_queens = df_queens[['RPT_DT', 'LAW_CAT_CD']]
df_queens = df_queens.dropna()

df_queens['RPT_DT'] = pd.to_datetime(df_queens['RPT_DT'])
df_queens['RPT_DT'] = df_queens['RPT_DT'].map(dt.datetime.toordinal)

x_values = sorted(df_queens['RPT_DT'].unique())

# Setting up dictionaries
violation_date_count_dict = {}
misdemeanor_date_count_dict = {}
felony_date_count_dict = {}

for value in x_values:
    violation_date_count_dict[value] = 0
    misdemeanor_date_count_dict[value] = 0
    felony_date_count_dict[value] = 0

for index, row in df_queens.iterrows():
    if row['LAW_CAT_CD'] == 'VIOLATION':
        violation_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'MISDEMEANOR':
        misdemeanor_date_count_dict[row['RPT_DT']] += 1
    elif row['LAW_CAT_CD'] == 'FELONY':
        felony_date_count_dict[row['RPT_DT']] += 1

violation_ys = []
misdemeanor_ys = []
felony_ys = []
for key, value in violation_date_count_dict.items():
    violation_ys.append(value)
for key, value in misdemeanor_date_count_dict.items():
    misdemeanor_ys.append(value)
for key, value in felony_date_count_dict.items():
    felony_ys.append(value)

independentVariables = pd.DataFrame(x_values, columns =['Date'])
dependentVariables_v = pd.DataFrame(violation_ys, columns=['Violations'])
dependentVariables_m = pd.DataFrame(misdemeanor_ys, columns=['Misdemeanors'])
dependentVariables_f = pd.DataFrame(felony_ys, columns=['Felonies'])

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(independentVariables, dependentVariables_v, test_size=0.2)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(independentVariables, dependentVariables_m, test_size=0.2)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(independentVariables, dependentVariables_f, test_size=0.2)

x_axis_v = X_test_v['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_m = X_test_m['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_f = X_test_f['Date'].map(dt.datetime.fromordinal).values.flatten()

plt.title(label="Queens")
X_v = X_test_v.values.flatten()
y_v = y_test_v.values.flatten()
z_v = np.polyfit(X_v, y_v, global_m_value)
p_v = np.poly1d(z_v)
plt.plot(x_axis_v, p_v(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten()
y_m = y_test_m.values.flatten()
z_m = np.polyfit(X_m, y_m, global_m_value)
p_m = np.poly1d(z_m)
plt.plot(x_axis_m, p_m(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten()
y_f = y_test_f.values.flatten()
z_f = np.polyfit(X_f, y_f, global_m_value)
p_f = np.poly1d(z_f)
plt.plot(x_axis_f, p_f(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()

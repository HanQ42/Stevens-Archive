import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.inspection import plot_partial_dependence

global_m_value = 3


"""Bronx"""
df_bronx = pd.read_csv('/Users/hanqingliu/Downloads/NYPD_ALL.csv', low_memory=False)
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

# # MSE testing
# # Depth
# depths = [6]
# MSE_Vio_train = []
# MSE_Vio_test = []
# MSE_Mis_train = []
# MSE_Mis_test = []
# MSE_Fel_train = []
# MSE_Fel_test = []
#
# accuracy_Vio_train = 0
# accuracy_Vio_test = 0
# accuracy_Mis_train = 0
# accuracy_Mis_test = 0
# accuracy_Fel_train = 0
# accuracy_Fel_test = 0
#
# for depth in depths:
#     # Vio
#     X_v_train = X_train_v.values.flatten().reshape(-1, 1)
#     X_v_test = X_test_v.values.flatten().reshape(-1, 1)
#     y_v_train = y_train_v.values.flatten()
#     y_v_test = y_test_v.values.flatten()
#     tree_regr_v_train = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_v_train, y_v_train)
#     y_v_pred_train = tree_regr_v_train.predict(X_v_train)
#     y_v_pred_test = tree_regr_v_train.predict(X_v_test)
#     MSE_Vio_train.append(mean_squared_error(y_v_pred_train, y_v_train))
#     MSE_Vio_test.append(mean_squared_error(y_v_pred_test, y_v_test))
#     # Mis
#     X_m_train = X_train_m.values.flatten().reshape(-1, 1)
#     X_m_test = X_test_m.values.flatten().reshape(-1, 1)
#     y_m_train = y_train_m.values.flatten()
#     y_m_test = y_test_m.values.flatten()
#     tree_regr_m_train = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_m_train, y_m_train)
#     y_m_pred_train = tree_regr_m_train.predict(X_m_train)
#     y_m_pred_test = tree_regr_m_train.predict(X_m_test)
#     MSE_Mis_train.append(mean_squared_error(y_m_pred_train, y_m_train))
#     MSE_Mis_test.append(mean_squared_error(y_m_pred_test, y_m_test))
#     # Fel
#     X_f_train = X_train_f.values.flatten().reshape(-1, 1)
#     X_f_test = X_test_f.values.flatten().reshape(-1, 1)
#     y_f_train = y_train_f.values.flatten()
#     y_f_test = y_test_f.values.flatten()
#     tree_regr_f_train = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_f_train, y_f_train)
#     y_f_pred_train = tree_regr_f_train.predict(X_f_train)
#     y_f_pred_test = tree_regr_f_train.predict(X_f_test)
#     MSE_Fel_train.append(mean_squared_error(y_f_pred_train, y_f_train))
#     MSE_Fel_test.append(mean_squared_error(y_f_pred_test, y_f_test))
#
#
# print(accuracy_Vio_train, accuracy_Vio_test, accuracy_Mis_train, accuracy_Mis_test, accuracy_Fel_train, accuracy_Fel_test)
#
# plt.title(label="MSE Vs. Depth for Violation")
# plt.xlabel('Depths')
# plt.ylabel('MSE')
# plt.plot(depths, MSE_Vio_train, 'b-', label="MSE for training data")
# plt.plot(depths, MSE_Vio_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()
#
# plt.title(label="MSE Vs. Depth for Misdemeanor")
# plt.xlabel('Depths')
# plt.ylabel('MSE')
# plt.plot(depths, MSE_Mis_train, 'b-', label="MSE for training data")
# plt.plot(depths, MSE_Mis_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()
#
# plt.title(label="MSE Vs. Depth for Felony")
# plt.xlabel('Depths')
# plt.ylabel('MSE')
# plt.plot(depths, MSE_Fel_train, 'b-', label="MSE for training data")
# plt.plot(depths, MSE_Fel_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()

# # leave node
# nodes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# MSE_Vio_train = []
# MSE_Vio_test = []
# MSE_Mis_train = []
# MSE_Mis_test = []
# MSE_Fel_train = []
# MSE_Fel_test = []
# for node in nodes:
#     # Vio
#     X_v_train = X_train_v.values.flatten().reshape(-1, 1)
#     X_v_test = X_test_v.values.flatten().reshape(-1, 1)
#     y_v_train = y_train_v.values.flatten()
#     y_v_test = y_test_v.values.flatten()
#     tree_regr_v_train = DecisionTreeRegressor(max_leaf_nodes=node).fit(X_v_train, y_v_train)
#     MSE_Vio_train.append(mean_squared_error(tree_regr_v_train.predict(X_v_train), y_v_train))
#     MSE_Vio_test.append(mean_squared_error(tree_regr_v_train.predict(X_v_test), y_v_test))
#     # Mis
#     X_m_train = X_train_m.values.flatten().reshape(-1, 1)
#     X_m_test = X_test_m.values.flatten().reshape(-1, 1)
#     y_m_train = y_train_m.values.flatten()
#     y_m_test = y_test_m.values.flatten()
#     tree_regr_m_train = DecisionTreeRegressor(max_leaf_nodes=node).fit(X_m_train, y_m_train)
#     MSE_Mis_train.append(mean_squared_error(tree_regr_m_train.predict(X_m_train), y_m_train))
#     MSE_Mis_test.append(mean_squared_error(tree_regr_m_train.predict(X_m_test), y_m_test))
#     # Fel
#     X_f_train = X_train_f.values.flatten().reshape(-1, 1)
#     X_f_test = X_test_f.values.flatten().reshape(-1, 1)
#     y_f_train = y_train_f.values.flatten()
#     y_f_test = y_test_f.values.flatten()
#     tree_regr_f_train = DecisionTreeRegressor(max_leaf_nodes=node).fit(X_f_train, y_f_train)
#     MSE_Fel_train.append(mean_squared_error(tree_regr_f_train.predict(X_f_train), y_f_train))
#     MSE_Fel_test.append(mean_squared_error(tree_regr_f_train.predict(X_f_test), y_f_test))
#
# plt.title(label="MSE Vs. Leaf Nodes for Violation")
# plt.xlabel('Leaf Nodes')
# plt.ylabel('MSE')
# plt.plot(nodes, MSE_Vio_train, 'b-', label="MSE for training data")
# plt.plot(nodes, MSE_Vio_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()
#
# plt.title(label="MSE Vs. Leaf Nodes for Misdemeanor")
# plt.xlabel('Leaf Nodes')
# plt.ylabel('MSE')
# plt.plot(nodes, MSE_Mis_train, 'b-', label="MSE for training data")
# plt.plot(nodes, MSE_Mis_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()
#
# plt.title(label="MSE Vs. Leaf Nodes for Felony")
# plt.xlabel('Leaf Nodes')
# plt.ylabel('MSE')
# plt.plot(nodes, MSE_Fel_train, 'b-', label="MSE for training data")
# plt.plot(nodes, MSE_Fel_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()



#
# for n in m:
#     temp_X_m = X_test_m.values.flatten()
#     temp_y_m = y_test_m.values.flatten()
#     temp_z_m = np.polyfit(temp_X_m, temp_y_m, n)
#     temp_p_m = np.poly1d(temp_z_m)
#     MSE_Mis.append(np.sum(np.square((y_test_m - temp_p_m(X_test_v)))) / (len(X_test_m)))
#
# plt.title(label="m")
# x = m
# y = MSE_Mis
# plt.xlabel = "m"
# plt.ylabel = "MSE"
# plt.plot(x, y, 'b-')
# plt.show()
#
# for n in m:
#     temp_X_f = X_test_f.values.flatten()
#     temp_y_f = y_test_f.values.flatten()
#     temp_z_f = np.polyfit(temp_X_f, temp_y_f, n)
#     temp_p_f = np.poly1d(temp_z_f)
#     MSE_Fel.append(np.sum(np.square((y_test_f - temp_p_f(X_test_f)))) / (len(X_test_f)))
#
# plt.title(label="m")
# x = m
# y = MSE_Fel
# plt.xlabel = "m"
# plt.ylabel = "MSE"
# plt.plot(x, y, 'r-')
# plt.show()
#
x_axis_v = X_test_v['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_m = X_test_m['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_f = X_test_f['Date'].map(dt.datetime.fromordinal).values.flatten()

plt.title(label="Bronx")
X_v = X_test_v.values.flatten().reshape(-1, 1)
y_v = y_test_v.values.flatten()
tree_regr_v = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_v, y_v)
plt.plot(x_axis_v, tree_regr_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
tree_regr_m = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_m, y_m)
plt.plot(x_axis_m, tree_regr_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
tree_regr_f = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_f, y_f)
plt.plot(x_axis_f, tree_regr_f.predict(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Manhattan"""
df_manhattan = pd.read_csv('/Users/hanqingliu/Downloads/Manhattan.csv', low_memory=False)
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
X_v = X_test_v.values.flatten().reshape(-1, 1)
y_v = y_test_v.values.flatten()
tree_regr_v = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_v, y_v)
plt.plot(x_axis_v, tree_regr_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
tree_regr_m = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_m, y_m)
plt.plot(x_axis_m, tree_regr_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
tree_regr_f = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_f, y_f)
plt.plot(x_axis_f, tree_regr_f.predict(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Staten Island"""
df_staten_island = pd.read_csv('/Users/hanqingliu/Downloads/Staten_Island.csv', low_memory=False)
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
X_v = X_test_v.values.flatten().reshape(-1, 1)
y_v = y_test_v.values.flatten()
tree_regr_v = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_v, y_v)
plt.plot(x_axis_v, tree_regr_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
tree_regr_m = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_m, y_m)
plt.plot(x_axis_m, tree_regr_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
tree_regr_f = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_f, y_f)
plt.plot(x_axis_f, tree_regr_f.predict(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Brooklyn"""
df_brooklyn = pd.read_csv('/Users/hanqingliu/Downloads/Brooklyn.csv', low_memory=False)
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
X_v = X_test_v.values.flatten().reshape(-1, 1)
y_v = y_test_v.values.flatten()
tree_regr_v = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_v, y_v)
plt.plot(x_axis_v, tree_regr_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
tree_regr_m = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_m, y_m)
plt.plot(x_axis_m, tree_regr_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
tree_regr_f = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_f, y_f)
plt.plot(x_axis_f, tree_regr_f.predict(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()


"""Queens"""
df_queens = pd.read_csv('/Users/hanqingliu/Downloads/Queens.csv', low_memory=False)
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
X_v = X_test_v.values.flatten().reshape(-1, 1)
y_v = y_test_v.values.flatten()
tree_regr_v = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_v, y_v)
plt.plot(x_axis_v, tree_regr_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
tree_regr_m = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_m, y_m)
plt.plot(x_axis_m, tree_regr_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
tree_regr_f = DecisionTreeRegressor(max_depth=6, max_leaf_nodes=40).fit(X_f, y_f)
plt.plot(x_axis_f, tree_regr_f.predict(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()
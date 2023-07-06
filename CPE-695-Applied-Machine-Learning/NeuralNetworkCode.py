import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.inspection import plot_partial_dependence

global_m_value = 3


"""Bronx"""
df_bronx = pd.read_csv('/Users/hanqingliu/Downloads/Bronx.csv', low_memory=False)
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
#
# # MSE testing
# # Iterations
# hidden_layers = [60]
# MSE_Vio_train = []
# MSE_Vio_test = []
# MSE_Mis_train = []
# MSE_Mis_test = []
# MSE_Fel_train = []
# MSE_Fel_test = []
# for size in hidden_layers:
#     # Vio
#     X_v_train = X_train_v.values.flatten()
#     X_v_test = X_test_v.values.flatten()
#     y_v_train = y_train_v.values.flatten()
#     y_v_test = y_test_v.values.flatten()
#     z_v_train = np.polyfit(X_v_train, y_v_train, 3)
#     p_v_train = np.poly1d(z_v_train)
#     MSE_Vio_train.append(mean_squared_error(p_v_train(X_v_train), y_v_train))
#     MSE_Vio_test.append(mean_squared_error(p_v_train(X_v_test), y_v_test))
#     # Mis
#     X_m_train = X_train_m.values.flatten()
#     X_m_test = X_test_m.values.flatten()
#     y_m_train = y_train_m.values.flatten()
#     y_m_test = y_test_m.values.flatten()
#     z_m_train = np.polyfit(X_m_train, y_m_train, 3)
#     p_m_train = np.poly1d(z_m_train)
#     MSE_Mis_train.append(mean_squared_error(p_m_train(X_m_train), y_m_train))
#     MSE_Mis_test.append(mean_squared_error(p_m_train(X_m_test), y_m_test))
#     # Fel
#     X_f_train = X_train_f.values.flatten()
#     X_f_test = X_test_f.values.flatten()
#     y_f_train = y_train_f.values.flatten()
#     y_f_test = y_test_f.values.flatten()
#     z_f_train = np.polyfit(X_f_train, y_f_train, 3)
#     p_f_train = np.poly1d(z_f_train)
#     MSE_Fel_train.append(mean_squared_error(p_f_train(X_f_train), y_f_train))
#     MSE_Fel_test.append(mean_squared_error(p_f_train(X_f_test), y_f_test))
#
# print(MSE_Vio_train, MSE_Vio_test, MSE_Mis_train, MSE_Mis_test, MSE_Fel_train, MSE_Fel_test)
# plt.title(label="MSE Vs. Hidden Layer Sizes for Violation")
# plt.xlabel('Hidden Layer Sizes')
# plt.ylabel('MSE')
# plt.plot(hidden_layers, MSE_Vio_train, 'b-', label="MSE for training data")
# plt.plot(hidden_layers, MSE_Vio_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()
#
# plt.title(label="MSE Vs. Hidden Layer Sizes for Misdemeanor")
# plt.xlabel('Hidden Layer Sizes')
# plt.ylabel('MSE')
# plt.plot(hidden_layers, MSE_Mis_train, 'b-', label="MSE for training data")
# plt.plot(hidden_layers, MSE_Mis_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()
#
# plt.title(label="MSE Vs. Hidden Layer Sizes for Felony")
# plt.xlabel('Hidden Layer Sizes')
# plt.ylabel('MSE')
# plt.plot(hidden_layers, MSE_Fel_train, 'b-', label="MSE for training data")
# plt.plot(hidden_layers, MSE_Fel_test, 'y-', label="MSE for testing data")
# plt.legend()
# plt.show()



x_axis_v = X_test_v['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_m = X_test_m['Date'].map(dt.datetime.fromordinal).values.flatten()
x_axis_f = X_test_f['Date'].map(dt.datetime.fromordinal).values.flatten()

plt.title(label="Bronx")

X_v = X_test_v.values.flatten().reshape(-1, 1)
y_v = y_test_v.values.flatten()
regressor_v = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=500).fit(X_v, y_v)
plt.plot(x_axis_v, regressor_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
regressor_m = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_m, y_m)
plt.plot(x_axis_m, regressor_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
regressor_f = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_f, y_f)
plt.plot(x_axis_f, regressor_f.predict(X_f), 'ro', label='Felonies')

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
regressor_v = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=500).fit(X_v, y_v)
plt.plot(x_axis_v, regressor_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
regressor_m = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_m, y_m)
plt.plot(x_axis_m, regressor_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
regressor_f = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_f, y_f)
plt.plot(x_axis_f, regressor_f.predict(X_f), 'ro', label='Felonies')

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
regressor_v = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=500).fit(X_v, y_v)
plt.plot(x_axis_v, regressor_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
regressor_m = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_m, y_m)
plt.plot(x_axis_m, regressor_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
regressor_f = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_f, y_f)
plt.plot(x_axis_f, regressor_f.predict(X_f), 'ro', label='Felonies')

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
regressor_v = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=500).fit(X_v, y_v)
plt.plot(x_axis_v, regressor_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
regressor_m = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_m, y_m)
plt.plot(x_axis_m, regressor_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
regressor_f = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_f, y_f)
plt.plot(x_axis_f, regressor_f.predict(X_f), 'ro', label='Felonies')

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
regressor_v = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=500).fit(X_v, y_v)
plt.plot(x_axis_v, regressor_v.predict(X_v), 'go', label='Violations')

X_m = X_test_m.values.flatten().reshape(-1, 1)
y_m = y_test_m.values.flatten()
regressor_m = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_m, y_m)
plt.plot(x_axis_m, regressor_m.predict(X_m), 'bo', label='Misdemeanors')

X_f = X_test_f.values.flatten().reshape(-1, 1)
y_f = y_test_f.values.flatten()
regressor_f = MLPRegressor(hidden_layer_sizes=[60, ],  activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.05, max_iter=200).fit(X_f, y_f)
plt.plot(x_axis_f, regressor_f.predict(X_f), 'ro', label='Felonies')

plt.legend()
plt.show()
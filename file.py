import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import re

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from time import time

# Считываем данные о min и max температурах
TEMP = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TAVG', 'PRCP', 'TMAX', 'TMIN'])
TEMP_MAX = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TMAX'])
TEMP_MIN = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TMIN'])

# Считываем данные о min и max температурах
TEMP = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TAVG', 'PRCP', 'TMAX', 'TMIN'])
TEMP_MAX = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TMAX'])
TEMP_MIN = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TMIN'])

# Функция, которая разбивает выборку по датам, напрмиер (1 января за все года)

def get_df_with_day_and_month_for_all_year (df, field_name):
    data = pd.DataFrame()
    for day in range(1, 32):
        for month in range (1, 13):
            if day < 10 and month < 10:
                pattern = r'\d{4}-'+f'0{month}-'+f'0{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)
                    pd.DataFrame(tmp).to_csv(f'Preprocessing/0{month}-0{day}.csv')
            if day < 10 and month > 9:
                pattern = r'\d{4}-'+f'{month}-'+f'0{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)
                    pd.DataFrame(tmp).to_csv(f'Preprocessing/{month}-0{day}.csv')
            if day > 9 and month < 10:
                pattern = r'\d{4}-'+f'0{month}-'+f'{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)
                    pd.DataFrame(tmp).to_csv(f'Preprocessing/0{month}-{day}.csv')
            if day > 9 and month > 9:
                pattern = r'\d{4}-'+f'{month}-'+f'{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)
                    pd.DataFrame(tmp).to_csv(f'Preprocessing/{month}-{day}.csv')
    return data


# Функция для удоавления выбросов в столбце

def delete_outliers (df):
    q = df.quantile([0.25, 0.75])
    # Межквартильное растояние
    low = q[0.25] - 1.5 * (q[0.75] - q[0.25])
    high = q[0.75] + 1.5 * (q[0.75] - q[0.25])
    return df[df.between(low, high)]

# Отрисовка boxplot

def draw_box_plot (df, plot_name):
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'Boxplot for {df.name}', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    _, bp = pd.DataFrame.boxplot(df, return_type='both', grid=False,  fontsize=15, figsize=(7,7))
    ax.set_ylabel(f'{plot_name}, $^o C$')
    plt.show()


t = get_df_with_day_and_month_for_all_year(TEMP_MIN, 'TMIN')


t1 = t.apply(lambda x: delete_outliers(x))
t1 = t1.dropna()
# t1.info

X, y = t1.T.iloc[:, :-1].values, t1.T.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)
prediction1 = regressor.predict(X_test)


print("The Explained Variance: %.2f" % regressor.score(X_train, y_train))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction1))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction1))

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

from sklearn.svm import SVR
#Fitting the Classifier
SVR = SVR(C=10.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)

SVR.fit(X_train_std, y_train)

prediction = SVR.predict(X_test_std)
print("SVM")
print('The Explained Variance: %.2f' % SVR.score(X_train_std, y_train))
print('The Mean Absolute Error: %.2f degrees celcius' % mean_absolute_error(
    y_test, prediction))
print('The Median Absolute Error: %.2f degrees celcius' %
      median_absolute_error(y_test, prediction))

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler(feature_range=(0,1))
X_train_n = ms.fit_transform(X_train)
X_test_n = ms.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(3,), max_iter=5000,
                   activation='logistic', solver='adam', alpha=0.001,
                           random_state=1, learning_rate_init=0.01)
mlp.fit(X_train_n, y_train)
prediction_mlp = mlp.predict(X_test_n)

print('\nMLP')
print("The Explained Variance: %.2f" % mlp.score(X_train_n, y_train))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction_mlp))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction_mlp))

from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()
clf = clf.fit(X_train_n, y_train)
prediction_DT = clf.predict(X_test_n)

print('\nDecision Tree')
print("The Explained Variance: %.2f" % mlp.score(X_train_n, y_train))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction_DT))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction_DT))

from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression

regr = AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
                         n_estimators=100, random_state=0)
regr.fit(X_train_std, y_train)

prediction_Ada_Boost = regr.predict(X_test_std)

print('\nAda Boost')
print("The Explained Variance: %.2f" % regr.score(X_train_std, y_train))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction_Ada_Boost))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction_Ada_Boost))


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

random_forest = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=100)
random_forest.fit(X_train, y_train)

predictiom_RF = random_forest.predict(X_test)

print('\nRandom Forest')
print("The Explained Variance: %.2f" % random_forest.score(X_train, y_train))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, predictiom_RF))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, predictiom_RF))


df = pd.DataFrame({'TEMP': y_test, 'TEMP_LR': prediction1, 'TEMP_SCR': prediction, 'TEMP_MLP': prediction_mlp, 'TEMP_DT': prediction_DT, 'TEMP_Ada_Boost': prediction_mlp})

df.plot(y = ['TEMP', 'TEMP_SCR', 'TEMP_MLP', 'TEMP_Ada_Boost'],color = ["dodgerblue", 'r', 'y', 'g', 'darkmagenta', 'pink'], style = ['', '--', '-', '-.', ''],linewidth=1, figsize=(25,10), )
plt.grid(True)
plt.xticks(rotation=75)
plt.xlabel('Date')
plt.ylabel('Max temperature, $^o, C$')
plt.title('Предсказание  максимальной температуры', fontsize=14, fontweight='bold')
plt.show()


from sklearn.ensemble import VotingRegressor

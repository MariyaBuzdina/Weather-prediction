import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error,  mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def rename_cols_rows(data, field_name):
    date = data['DATE'].apply(pd.to_datetime, errors='ignore')
    year = date.map(lambda x: x.year).values
    day_month = date.map(lambda x:x.strftime('%m-%d')).unique()[0]
    return data.drop('DATE', 1).set_index(year).rename(columns={f'{field_name}': f'{day_month}'})

# Функция, которая разбивает выборку по датам, напрмиер (1 января за все года)

def get_df_with_day_and_month_for_all_year (df, field_name):
    data = pd.DataFrame()
    for month in range (1, 13):
        for day in range(1, 32):
            if day < 10 and month < 10:
                pattern = r'\d{4}-'+f'0{month}-'+f'0{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    pd.DataFrame(tmp).to_csv(f'{field_name}/0{month}-0{day}.csv')
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)

            if day < 10 and month > 9:
                pattern = r'\d{4}-'+f'{month}-'+f'0{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    pd.DataFrame(tmp).to_csv(f'{field_name}/{month}-0{day}.csv')
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)

            if day > 9 and month < 10:
                pattern = r'\d{4}-'+f'0{month}-'+f'{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    pd.DataFrame(tmp).to_csv(f'{field_name}/0{month}-{day}.csv')
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)
                    pd.DataFrame(tmp).to_csv(f'{field_name}/0{month}-{day}.csv')
            if day > 9 and month > 9:
                pattern = r'\d{4}-'+f'{month}-'+f'{day}'
                tmp = df[df['DATE'].str.match(pattern)]
                if not tmp.empty:
                    pd.DataFrame(tmp).to_csv(f'{field_name}/{month}-{day}.csv')
                    tmp = rename_cols_rows(tmp, field_name)
                    data = pd.concat([data, tmp], axis = 1, join='outer', sort=True)

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


#Создание тренировочного и тестового набора для проверки работы моделей
def create_train_and_test_datasets (data):
    data = data.dropna()
#     apply(lambda x: delete_outliers(x)).dropna()
    X, y = data.T.iloc[:, :-1].values, data.T.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # X_train, X_test, y_train, y_test = np.array(X[:-40]), np.array(X[-40:]), np.array(y[:-40]), np.array(y[-40:])
    return X_train, X_test, y_train, y_test

# Стандартизация datasets для использования в моеделях
def standartization_dataset (X_train, X_test):
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    return X_train_std, X_test_std

# Нормализация
def normalization_dataset (X_train, X_test):
    ms = MinMaxScaler(feature_range=(0,1))
    X_train_n = ms.fit_transform(X_train)
    X_test_n = ms.transform(X_test)
    return X_train_n, X_test_n

# Отрисовка столбчатых диаграм для оценки работы моделей
def draw_bar(df, field_name):
    df.plot.bar(color = ["dodgerblue", 'r', 'darkmagenta', 'c'], figsize=(7,7))
    plt.xticks(rotation=75)
    plt.ylabel(f'{field_name}')
    plt.title(f'{field_name}', fontsize=14, fontweight='bold')
    plt.show()

# Полученеие данных о модели
def estimate_model (model, X_train, y_train, y_test, prediction, model_name):
    score = model.score(X_train, y_train)
    MAE = mean_absolute_error(y_test, prediction)
    MSE = mean_squared_error(y_test, prediction)
    print(f'\n{model_name}')
    print("The Explained Variance: %.2f" % score)
    print("The Mean Absolute Error: %.2f degrees celsius" % MAE)
    print('The Square Error: %.2f' % MSE)
    return pd.DataFrame({"Score": score, 'MAE': MAE, 'MSE': MSE}, index=[f'{model_name}'])


def reserch_prediction(data, field_name):
    X_train, X_test, y_train, y_test = create_train_and_test_datasets(data)
    X_train_std, X_test_std = standartization_dataset(X_train, X_test)
    X_train_n, X_test_n = normalization_dataset(X_train, X_test)

    LR = LinearRegression()
    LR.fit(X_train, y_train)
    prediction_LR = LR.predict(X_test)
    estimate_LR = estimate_model(LR, X_train, y_train, y_test, prediction_LR, f"LR {field_name}")

    svr = SVR(C=100.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)
    svr.fit(X_train_std, y_train)
    prediction_SVM = svr.predict(X_test_std)
    estimate_SVM = estimate_model(svr, X_train_std, y_train, y_test, prediction_SVM, f"SVM {field_name}")

    mlp = MLPRegressor(hidden_layer_sizes=(3,), max_iter=5000,
                       activation='relu', solver='adam', alpha=0.001,
                       random_state=1, learning_rate_init=0.01)
    mlp.fit(X_train_n, y_train)
    prediction_mlp = mlp.predict(X_test_n)
    estimate_MLP = estimate_model(mlp, X_train_n, y_train, y_test, prediction_mlp, f"MLP {field_name}")

    random_forest = RandomForestRegressor(max_depth=10, random_state=0,
                                          n_estimators=1000)
    random_forest.fit(X_train, y_train)
    prediction_RF = random_forest.predict(X_test)
    estimate_RF = estimate_model(random_forest, X_train, y_train, y_test, prediction_RF, f"RF {field_name}")

    svm = SVR(C=100.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.001, verbose=False)

    rf = RandomForestRegressor(max_depth=2, random_state=0,
                               n_estimators=100)

    er = VotingRegressor([('svm', svm), ('rf', rf)])
    er.fit(X_train_std, y_train)
    pred = er.predict(X_test_std)
    estimate_ens = estimate_model(er, X_train_std, y_train, y_test, pred, f"EM {field_name}")

    estimate = pd.concat([estimate_LR, estimate_SVM, estimate_MLP, estimate_RF, estimate_ens], axis=0, join='outer', sort=True)

    df = pd.DataFrame({'TEMP': y_test, 'TEMP_LR': prediction_LR, 'TEMP_SVM': prediction_SVM, 'TEMP_MLP': prediction_mlp,
                       'TEMP_RF': prediction_RF, 'EnsM': pred})
    df.plot(y=['TEMP', 'TEMP_SVM', 'TEMP_MLP', 'TEMP_RF', 'EnsM'], color=["dodgerblue", 'r', 'darkmagenta', 'c', 'y'],
            style=['', '--', ':', '-.', ''], linewidth=1, figsize=(25, 10), )
    plt.grid(True)
    plt.xticks(rotation=75)
    plt.xlabel('Date')
    plt.ylabel(f'{field_name} temperature, $^o, C$')
    plt.title(f'Предсказание {field_name} температуры', fontsize=14, fontweight='bold')
    plt.show()

    return df, estimate

# Считываем данные о min и max температурах
TEMP_MAX = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TMAX'])
TEMP_MIN = pd.read_csv('input/Saint-Petersburg(1881).csv', delimiter=',', usecols = ['DATE', 'TMIN'])
t_max = get_df_with_day_and_month_for_all_year(TEMP_MAX, 'TMAX')
t_min = get_df_with_day_and_month_for_all_year(TEMP_MIN, 'TMIN')
draw_box_plot(t_max['03-26'], 'TMAX')
draw_box_plot(t_min['03-26'], 'TMIN')
X_train_max, X_test_max, y_train_max, y_test_max = create_train_and_test_datasets(t_max)
X_train_min, X_test_min, y_train_min, y_test_min = create_train_and_test_datasets(t_min)

df_max, estimate_max = reserch_prediction(t_max, 'MAX')
draw_bar(estimate_max['Score'], 'Коэффициент детерминизации')
draw_bar(estimate_max['MAE'], 'MAE')
draw_bar(estimate_max['MSE'], 'MSE')

df_min, estimate_min = reserch_prediction(t_min, 'MIN')
draw_bar(estimate_min['Score'], 'Коэффициент детерминизации')
draw_bar(estimate_min['MAE'], 'MAE')
draw_bar(estimate_min['MSE'], 'MSE')


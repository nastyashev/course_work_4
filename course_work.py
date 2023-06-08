import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)


def replace_with_unique_numbers(data, columns):
    for column in columns:
        data[column], _ = pd.factorize(data[column])
    return data


if __name__ == "__main__":
    d_path = r'C:\Users\nasty\OneDrive - kpi.ua\KPI\курсова 4 сем\CrabAgePrediction.csv'
    data_frame = pd.read_csv(d_path, sep=',', decimal='.')

    # аналіз початкового датасету
    print('\nІнформація про датасет:')
    data_frame.info()
    print('\nПерші 5 рядків:')
    print(data_frame.head())
    print('\nКолонки з пропущеними значеннями:')
    print(data_frame.isna().any())
    print('\nОпис датасету:')
    print(data_frame.describe())

    # заміняємо колони зі строковими значеннями на числові
    columns_to_replace = ['Sex']
    data_frame = replace_with_unique_numbers(data_frame, columns_to_replace)
    print('\nПерші 5 рядків:')
    print(data_frame.head())

    # розділення на навчальну і тестову вибірки
    x = data_frame.drop(columns='Age')
    y = data_frame['Age']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # побудова моделей

    # лінійна регресія
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    y_pred_linear = linear_reg.predict(x_test)

    mse_lin = mean_squared_error(y_test, y_pred_linear)
    r2_lin = r2_score(y_test, y_pred_linear)
    print('\nLinear Regression')
    print('MSE:', mse_lin)
    print('R^2:', r2_lin, '\n')

    plt.scatter(y_test, y_pred_linear)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Фізичні дані')
    plt.ylabel('Вік')
    plt.title('Лінійна регресія')
    plt.show()

    # випадковий ліс
    random_forest_reg = RandomForestRegressor()
    random_forest_reg.fit(x_train, y_train)
    y_pred_rand_forest = random_forest_reg.predict(x_test)

    mse_rand_forest = mean_squared_error(y_test, y_pred_rand_forest)
    r2_rand_forest = r2_score(y_test, y_pred_rand_forest)
    print('Random Forest')
    print('MSE:', mse_rand_forest)
    print('R^2:', r2_rand_forest, '\n')

    plt.scatter(y_test, y_pred_rand_forest)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Фізичні дані')
    plt.ylabel('Вік')
    plt.title('Випадковий ліс')
    plt.show()

    # KNeighborsRegressor
    random_knn = KNeighborsRegressor()
    random_knn.fit(x_train, y_train)
    y_pred_knn = random_knn.predict(x_test)

    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)
    print('KNN')
    print('MSE:', mse_knn)
    print('R^2:', r2_knn, '\n')

    plt.scatter(y_test, y_pred_knn)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Фізичні дані')
    plt.ylabel('Вік')
    plt.title('KNN')
    plt.show()


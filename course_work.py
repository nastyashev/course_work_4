import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


pd.set_option('display.max_columns', None)


def replace_with_unique_numbers(data, columns):
    for column in columns:
        data[column], _ = pd.factorize(data[column])
    return data


if __name__ == "__main__":
    d_path = r'C:\Users\nasty\OneDrive - kpi.ua\KPI\курсова 4 сем\eurovision_1957-2021.csv'
    data_frame = pd.read_csv(d_path, sep=',', decimal='.')

    # аналіз початкового датасету
    # print('\nІнформація про датасет:')
    # data_frame.info()
    # print('\nПерші 5 рядків:')
    # print(data_frame.head())
    # print('\nКолонки з пропущеними значеннями:')
    # print(data_frame.isna().any())
    # print('\nОпис датасету:')
    # print(data_frame.describe())

    # виведення суми балів по роках
    # yearly_points = data_frame.groupby('Year')['Points'].sum()
    # plt.figure(figsize=(12, 6))
    # plt.plot(yearly_points.index, yearly_points.values)
    # plt.xlabel('Рік')
    # plt.ylabel('Сума балів')
    # plt.title('Залежність суми балів від року')
    # plt.xticks(ticks=yearly_points.index[::5], rotation=45)
    # plt.show()
    # бачимо що в 2016 сума змінилась, це вказуєна зміну системи голосування
    points_type_by_year = data_frame.groupby('Year')['Points type'].unique()
    if 'Points given' not in points_type_by_year[2016]:
        print("\nВ 2016 році змінився тип голосування.")
        print("Унікальні значення стовпця 'Points type' для 2016 року:", points_type_by_year[2016])

    # видаляємо всі записи зі старою системою голосування та колонки які не впливають на подальший аналіз
    data_frame = data_frame[data_frame['Year'] >= 2016]
    data_frame = data_frame.drop(columns=['Unnamed: 0', 'Edition', 'Year', 'Points type'])

    # заміняємо колони зі строковими значеннями на числові для зручності
    columns_to_replace = ['From', 'To']
    data_frame = replace_with_unique_numbers(data_frame, columns_to_replace)
    print('\nПерші 5 рядків:')
    print(data_frame.head())

    # розділення на навчальну і тестову вибірки
    x = data_frame.drop(columns='Points')
    y = data_frame['Points']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # побудова моделей

    # лінійна регресія
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    y_pred_linear = linear_reg.predict(x_test)

    mse_lin = mean_squared_error(y_test, y_pred_linear)
    r2_lin = r2_score(y_test, y_pred_linear)
    print('Linear Regression')
    print('MSE:', mse_lin)
    print('R^2:', r2_lin, '\n')

    plt.scatter(y_test, y_pred_linear)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Справжні бали')
    plt.ylabel('Прогнозовані бали')
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
    plt.xlabel('Справжні бали')
    plt.ylabel('Прогнозовані бали')
    plt.title('Випадковий ліс')
    plt.show()

    # Градієнтний бустінг
    random_grad_boost = GradientBoostingRegressor()
    random_grad_boost.fit(x_train, y_train)
    y_pred_grad_boost = random_grad_boost.predict(x_test)

    mse_grad_boost = mean_squared_error(y_test, y_pred_grad_boost)
    r2_grad_boost = r2_score(y_test, y_pred_grad_boost)
    print('Gradient Boosting')
    print('MSE:', mse_grad_boost)
    print('R^2:', r2_grad_boost, '\n')

    plt.scatter(y_test, y_pred_grad_boost)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Справжні бали')
    plt.ylabel('Прогнозовані бали')
    plt.title('Градієнтний бустінг')
    plt.show()

    # SVR
    random_svr = SVR()
    random_svr.fit(x_train, y_train)
    y_pred_svr = random_svr.predict(x_test)

    mse_svr = mean_squared_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    print('SVR')
    print('MSE:', mse_svr)
    print('R^2:', r2_svr, '\n')

    plt.scatter(y_test, y_pred_svr)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Справжні бали')
    plt.ylabel('Прогнозовані бали')
    plt.title('SVR')
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
    plt.xlabel('Справжні бали')
    plt.ylabel('Прогнозовані бали')
    plt.title('KNN')
    plt.show()

    # DecisionTreeRegressor
    random_dec_tree = DecisionTreeRegressor()
    random_dec_tree.fit(x_train, y_train)
    y_pred_dec_tree = random_dec_tree.predict(x_test)

    mse_dec_tree = mean_squared_error(y_test, y_pred_dec_tree)
    r2_dec_tree = r2_score(y_test, y_pred_dec_tree)
    print('DecisionTreeRegressor')
    print('MSE:', mse_dec_tree)
    print('R^2:', r2_dec_tree, '\n')

    plt.scatter(y_test, y_pred_dec_tree)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Справжні бали')
    plt.ylabel('Прогнозовані бали')
    plt.title('DecisionTreeRegressor')
    plt.show()


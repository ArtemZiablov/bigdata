import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Завантаження датасету з вказанням опції для уникнення змішаних типів
file_path = 'C:\\Users\\Artem\\Downloads\\Taiwan Air Quality Index Data 2016~2024\\air_quality.csv'
data = pd.read_csv(file_path, low_memory=False)

# Вибір числових колонок для PCA
numerical_columns = ['aqi', 'so2', 'co', 'o3', 'o3_8hr', 'pm10', 'pm2.5', 'no2', 'nox', 'no',
                     'windspeed', 'winddirec', 'co_8hr', 'pm2.5_avg', 'pm10_avg', 'so2_avg',
                     'longitude', 'latitude']

# Заміна некоректних значень (наприклад, '-') на NaN
data[numerical_columns] = data[numerical_columns].replace('-', np.nan)

# Перетворення колонок на числові значення, якщо вони такими не є
data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Видалення рядків з пропущеними значеннями
data = data.dropna(subset=numerical_columns)

# Стандартизація даних
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_columns])

# Застосування PCA (поки що для всіх компонент)
pca = PCA()
pca_transformed_data = pca.fit_transform(scaled_data)

# Побудова графіка кумулятивної поясненої дисперсії для визначення оптимальної кількості компонент
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Кумулятивна пояснена дисперсія залежно від кількості компонент')
plt.xlabel('Кількість компонент')
plt.ylabel('Кумулятивна пояснена дисперсія')
plt.grid(True)
plt.show()

# Пошук оптимальної кількості компонент на основі 80% поясненої дисперсії
optimal_components = (cumulative_variance >= 0.80).argmax() + 1  # Знаходимо оптимальну кількість компонент, в нашому випадку виходить 7
print(f'Оптимальна кількість цифрових компонент: {optimal_components}')

# Застосування PCA з оптимальною кількістю компонент
pca = PCA(n_components=optimal_components)  # за замовчуванням параметр svd_solver='auto'
pca_transformed_data = pca.fit_transform(scaled_data)

# Створення DataFrame для перетворених даних
pca_columns = [f'ПК{i+1}' for i in range(optimal_components)]
pca_df = pd.DataFrame(data=pca_transformed_data, columns=pca_columns)

# Додавання назад оригінальних нечислових колонок до перетворених даних
final_pca_df = pd.concat([data[['date', 'sitename', 'county', 'pollutant', 'status']], pca_df], axis=1)

# Виведення перших кількох рядків перетвореного датасету
print("Перші 5 рядків перетвореного датасету:")
print(final_pca_df.head())

# Побудова графіка порівняння кількості колонок до і після застосування PCA, включаючи всі колонки
columns_before = len(data.columns)  # Усі колонки до застосування PCA
columns_after = len(final_pca_df.columns)  # Усі колонки після застосування PCA, включаючи нечислові колонки

# Створення стовпчастої діаграми
plt.figure(figsize=(8, 6))
plt.bar(['До PCA', 'Після PCA'], [columns_before, columns_after], color=['blue', 'green'])
plt.title('Порівняння кількості стовпчиків до та після застосування PCA')
plt.ylabel('Кількість колонок')
plt.yticks(np.arange(0, columns_before + 1, 1))
plt.show()

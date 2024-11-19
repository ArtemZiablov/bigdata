import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Завантажуєсо набір даних Iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # беремо перші дві ознаки для спрощення візуалізації
y = iris.target

# Розділяється набір даних на тренувальну та тестову частини
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3561)

C_values = [0.1, 1, 10, 100]  # значення параметра регуляризації C
kernels = ['linear', 'rbf', 'poly']  # типи ядер
gamma_values = ['scale', 'auto']  # значення gamma
degrees = [2, 3, 4]  # ступені полінома


def plot_svm_results(kernel, C, gamma='scale', degree=3):
    # Створюємо модель SVM з відповідними параметрами
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    # Навчаємо модель
    model.fit(X_train, y_train)

    # Передбачаємо результати для тестової вибірки та оцінюємо точність
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Візуалізуюємо всі точки даних
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title(f'Kernel: {kernel}, C: {C}, Gamma: {gamma}, Degree: {degree}, Accuracy: {accuracy:.2f}')

    # Створюємо сітку для візуалізації меж класифікації
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)


# Перебираються різні ядра та параметри, щоб показати, як вони впливають на модель
for kernel in kernels:
    if kernel == 'poly':  # Для поліноміального ядра тестуємо різні ступені полінома
        plt.figure(figsize=(15, 15))
        for i, (C, degree) in enumerate(zip(C_values, degrees)):
            plt.subplot(2, 2, i + 1)
            plot_svm_results(kernel=kernel, C=C, degree=degree)
        plt.tight_layout()
        plt.show()
    else:  # Для інших ядер тестуються різні значення gamma
        for gamma in gamma_values:
            plt.figure(figsize=(15, 10))
            for i, C in enumerate(C_values):
                plt.subplot(2, 2, i + 1)
                plot_svm_results(kernel=kernel, C=C, gamma=gamma)
            plt.tight_layout()
            plt.show()

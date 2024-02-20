from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

#可视化训练集和测试集数据
def plot_train_test_data(training_target, testing_target, y_train_pred, y_test_pred):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=80)
    plt.rcParams['font.sans-serif'] = ['Arial']  # 字体均为 Arial
    plt.rcParams['axes.unicode_minus'] = False

    #设置图框粗细
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    #设置字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('E$_a$$_d$$_s$$^D$$^F$$^T$ (eV)', fontsize=20)
    plt.ylabel('E$_a$$_d$$_s$$^p$$^r$$^e$$^d$ (eV)', fontsize=20)
    #设置坐标轴刻度线朝内


    ax.tick_params(direction='in', length=10, width=2, colors='#000000', grid_color='#000000', grid_alpha=0.5)
    scatter1 = plt.scatter(x=training_target, y=y_train_pred, s=80, marker='s', c='blue', alpha=0.8, label='train data',
                           linewidths=0.3, edgecolor='#17223b')
    #设置marker为空心圆
    scatter2 = plt.scatter(x=testing_target, y=y_test_pred, s=80, marker='o', c='red', alpha=0.8, label='test data',
                           linewidths=0.3, edgecolor='#17223b')
    ax.plot([-9, 1], [-9, 1], '--', c='black', alpha=0.5)
    plt.legend(loc='upper left', fontsize=20, frameon=True, labelspacing=0.5)

    return plt



def plot_learning_curve1(estimator, x_train, x_test, y_train, y_test):
    train_sizes, train_scores, test_scores = learning_curve(estimator, x_train, y_train, cv=5, scoring='neg_mean_squared_error',
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation')
    plt.xlabel('Training examples')
    plt.ylabel('MSE')
    plt.legend(loc='best')
    plt.show()
    return plt

def plot_feature_importance(feature_importance_dict):
    feature_importance_dict = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    feature_name = []
    feature_importance = []
    for i in range(len(feature_importance_dict)):
        feature_name.append(feature_importance_dict[i][0])
        feature_importance.append(feature_importance_dict[i][1])
    plt.figure(figsize=(10, 10))
    plt.barh(range(len(feature_importance)), feature_importance, tick_label=feature_name)
    plt.title("Feature Importance")
    return plt
def plot_permutation_feature_importance(df):

    df = df.sort_values(by='weight', ascending=False)
    plt.figure(figsize=(10, 6))
    #设置横纵坐标标题大小
    plt.xlabel('weight', fontsize=20)
    plt.ylabel('feature', fontsize=20)

    sns.barplot(x='weight', y='feature', data=df)
    #"weight"和"feature"字体大小设置
    plt.tick_params(labelsize=17)

    plt.title('Feature Importance', fontsize=20)

    plt.tight_layout()
    return plt


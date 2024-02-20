from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
import pandas as pd
import sklearn.metrics
from utils import plot_learning_curve1, plot_train_test_data,plot_permutation_feature_importance
import eli5
from eli5.sklearn import PermutationImportance

#----------------------------------------------------------#
# 划分数据
#----------------------------------------------------------#
FOLD = 5
df = pd.read_csv(r'sorted_oxide.csv')
split = StratifiedKFold(FOLD, random_state=42, shuffle=True)

for k, (_, test_idx) in enumerate(split.split(df, df.metal)):
    df.loc[test_idx, 'split'] = k

#打印分组中每组的吸附质原子df.atom
print(df.groupby('split').atom_x.value_counts().unstack())

df.split = df.split.astype(int)

# 打印各个折里各氧化物的样本数量
print(df.groupby('split').metal.value_counts().unstack())

#记录各个折的分组情况,groups是一个字典，key是split的值，value是对应的index,index是数据中的行号
split_groups = df.groupby('split').groups
print(split_groups)

#打印每个折的oxide_ads列的值
# for i in range(FOLD):
#     print("第{}折".format(i))
#     print(df.loc[split_groups[i], 'oxide_ads'].unique())

#将df中site值为1和3的行保留，其他的去掉
df = df[df.site.isin([1, 3])]

#打印df长度
print(len(df))

# drop掉不需要的特征
df = df.drop(columns=['oxide','metal', 'atom_x','site','atom_y','oxide_atom'])
df=df.drop(columns=['bulk_VBM','bulk_CBM','slab_VBM','slab_CBM'])
df=df.drop(columns=['bulk_M_p_width','bulk_O_s_center','bulk_O_s_width','MO_a_HOMO','MO_b_HOMO','MO_a_LUMO','MO_b_LUMO','MO_a_gap','MO_b_gap','MO_gap']) #0.80 0.72 0.62 0.82
# df=df.drop(columns=['metal_ads'])
df=df.drop(columns=['M_VEN','test','EAO'])
df=df.drop(columns=['A_UPOEN','A_USOEN','M_SOEN','bulk_M_p_center','bulk_M_d_width','bulk_O_p_width','MO_AEA','MO_AIP','MO_HOMO','MO_LUMO','A_IC','M_AP','M_IR','A_IR','M_IC','M_AN','M_FEA','M_IP','M_ENC','A_AR','bulk_M_d_center','A_FIP','A_ICD','A_FIP','A_IP','M_FIP'])

#继续删减
# df=df.drop(columns=['NO','M_PE','metal_ads','A_ENC'])

#打印所有特征名称
print(df.columns)

#----------------------------------------------------------#
# 选择模型
#----------------------------------------------------------#

# GradientBoostingRegressor
# params = {
#     "n_estimators": 50,
#     "max_depth": 7,
#     "min_samples_split": 5,
#     "learning_rate": 0.25,
#     #alpha是GBR的正则化参数，subsample是每次迭代随机选择的样本比例
#     "alpha":0.95,
#     "max_features":0.25,
#     "min_samples_leaf":1,
#     "min_samples_split":5,
#     "subsample":0.65,
# }
# exported_pipeline = make_pipeline(GradientBoostingRegressor(**params))

# RFR 随机森林
# from sklearn.ensemble import RandomForestRegressor
# exported_pipeline = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, max_features=0.25, min_samples_leaf=1, random_state=42)

# SVR SVM
# from sklearn.svm import SVR
# exported_pipeline = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# MLPR MLP
# from sklearn.neural_network import MLPRegressor
# exported_pipeline = MLPRegressor(hidden_layer_sizes=(10, 7, 5), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

# ---automl搜出来的模型1, 过拟合较严重
# exported_pipeline = GradientBoostingRegressor(alpha=0.9, learning_rate=0.5, loss="squared_error", max_depth=7, max_features=0.1, min_samples_leaf=6, min_samples_split=19, n_estimators=100, subsample=0.9500000000000001)

# ---automl搜出来的模型2, split的随机种子取42
from sklearn.linear_model import LassoLarsCV
from xgboost import XGBRegressor
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=10, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.9000000000000001, verbosity=0)),
    LassoLarsCV(normalize=True)
)

# ---automl搜出来的模型3, split的随机种子取42
# exported_pipeline = make_pipeline(
#     StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=True, max_features=0.45, min_samples_leaf=5, min_samples_split=2, n_estimators=100)),
#     GradientBoostingRegressor(alpha=0.8, learning_rate=0.5, loss="ls", max_depth=10, max_features=0.8, min_samples_leaf=15, min_samples_split=3, n_estimators=100, subsample=1.0)
# )

#----------------------------------------------------------#
# k折训练
#----------------------------------------------------------#
for fold in range(FOLD):
    df_train = df[df.split != fold].drop(columns=['split'])
    df_test = df[df.split == fold].drop(columns=['split'])
    training_features, training_target = df_train.drop(columns=['oxide_ads']), df_train['oxide_ads']
    testing_features, testing_target = df_test.drop(columns=['oxide_ads']), df_test['oxide_ads']
    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
    perm = PermutationImportance(exported_pipeline, random_state=1).fit(training_features, training_target)  # 实例化

    # 打印每个特征的重要性得分
    importance_dataframe = eli5.format_as_dataframe(
        eli5.explain_weights(perm, feature_names=training_features.columns.tolist()))

    # 特征重要性画图
    plt = plot_permutation_feature_importance(importance_dataframe)
    plt.savefig('fold' + str(fold) + '_perm.png')

    # 打印MSE,MAE,R2
    print("MSE：", round(sklearn.metrics.mean_squared_error(testing_target, results, squared=False), 4))
    print("MAE：", round(sklearn.metrics.mean_absolute_error(testing_target, results), 4))
    print("R2：", round(sklearn.metrics.r2_score(testing_target, results), 4))

    # 画图
    train_res = exported_pipeline.predict(training_features)
    plt1 = plot_train_test_data(training_target, testing_target, train_res, results)
    plt1.savefig('fold' + str(fold) + '.png')
    # plt1.show()

    #学习曲线
    plt2=plot_learning_curve1(exported_pipeline, training_features, testing_features, training_target, testing_target)

    #打印离群点的序号与值
    print("df_test[abs(testing_target-results) > 2 离群点的oxide_ads值")
    print(df_test[abs(testing_target-results) > 2][['oxide_ads']])

#----------------------------------------------------------#
# 其他特征重要性分析
#----------------------------------------------------------#

# 使用mlxtend程序包分析特征重要性
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs = SFS(exported_pipeline,
              k_features=3,
                forward=True,
                floating=False,
                verbose=2,
                # scoring='neg_mean_absolute_error',
                scoring='r2',
                cv=5)

df = df.drop(columns=['split'])
training_features, training_target = df.drop(columns=['oxide_ads']), df['oxide_ads']

# 用这个方法评估重要性要重新训练模型
sfs = sfs.fit(training_features, training_target)

print(sfs.k_feature_names_)  # 模型换成RFR的话，前十重要的特征也差不多

feature_importances = sfs.get_metric_dict()
print(feature_importances)  # 打印的是每个特征子集及其得分

#打印所有特征的重要性得分
for i in range(1,len(feature_importances)+1):
    print(feature_importances[i]['feature_names'], feature_importances[i]['avg_score'])

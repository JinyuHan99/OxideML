import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 画出特征之间的关联情况
def corr_heatmap(data, map_name):
    corr = data.corr()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(10, 10))

    #设置特征名字体大小
    plt.rcParams['font.size'] = 9.5
    sns.heatmap(corr, annot=True, linewidths=0.25, vmax=1.0, square=True, cmap="Blues")

    #tight_layout可以自动调整子图参数，使之填充整个图像区域
    plt.tight_layout()

    # 保存图片
    fig.savefig(map_name, dpi=500)

    # 清除图片
    plt.close(fig)


# metal_features = pd.read_csv('metal_feature.csv').iloc[:, 1:]
# nonmetal_features = pd.read_csv('nonmetal_feature.csv').iloc[:, 1:]
oxide_bulk_feature=pd.read_csv('oxide_bulk_feature.csv').iloc[:, 1:]
#drop掉M_p_width列
# oxide_bulk_feature=oxide_bulk_feature.drop(['bulk_M_p_width','bulk_O_s_center','bulk_O_s_width','MO_a_HOMO','MO_b_HOMO','MO_a_LUMO','MO_b_LUMO','MO_a_gap','MO_b_gap','MO_gap'],axis=1)
# corr_heatmap(metal_features,'metal_corr_heatmap.png')
# corr_heatmap(nonmetal_features,'nonmetal_corr_heatmap.png')
corr_heatmap(oxide_bulk_feature,'oxide_bulk_corr_heatmap.png')
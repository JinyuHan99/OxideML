import pandas as pd
import matplotlib.pyplot as plt

#画出吸附能的分布情况
df = pd.read_csv(r'sorted_oxide.csv')

#对df中的oxide_ads列进行区间频数统计并绘制直方图以展现数据分布情况,每个区间的宽度为0.5
#构建从-9到0，间隔为0.5的区间
bins = []
num = -9.0
while num < 0:
    bins.append(num)
    num += 0.5
# print(bins)

#以bins为区间，统计df中的oxide_ads列的频数,alpha是透明度
df['oxide_ads'].hist(bins=bins,edgecolor='black',color='orange',alpha=0.8)

#横坐标范围为-9.5到0.5
plt.xlim(-9,0)

#去除网格线
plt.grid(False)

#line tick朝内
plt.tick_params(direction='in')

plt.xlabel('Adsorption energy (eV)')
plt.ylabel('Count')

# plt.show()
plt.tight_layout()

#保存图片
plt.savefig('freq.png',dpi=300)
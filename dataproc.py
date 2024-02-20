import pandas as pd


#数据预处理
#将oxide.csv中的数据读入到df中
df = pd.read_csv('oxide.csv')

#oxide.csv中的列包括“oxide”, "atom",和"site1","site2","site3","site4","site5","site6"
#将site1到site6的吸附能数据合并为一列，列名为ads，另外添加一列，列名为site，值为site1到site6
df = pd.melt(df, id_vars=['oxide', 'atom'], value_vars=['site1', 'site2', 'site3', 'site4', 'site5', 'site6'], var_name='site', value_name='oxide_ads')

#将site列中的site1到site6替换为1到6
df.site = df.site.str.replace('site', '').astype(int)

#将ads为NaN的行删除
df = df.dropna(subset=['oxide_ads'])

# 将oxide列和atom列合并为一列，列名为oxide_atom，内容oxide列的内容+ “+” + atom列的内容
df['oxide_atom'] = df.oxide + '+' + df.atom

#按照oxide_atom列的内容进行排序
df = df.sort_values('oxide_atom')

df_features = pd.read_csv(r'features.csv')

#将df_features中的metal列加上‘O+’+atom列的内容，得到新的列名为oxide_atom
df_features['oxide_atom'] =  df_features.metal+'O+'+df_features.atom

#将df_feature中oxide_atom列的内容与df中的oxide_atom列的内容进行匹配，将匹配成功的合并
df = pd.merge(df, df_features, on='oxide_atom', how='left')
# print(df)

#在df后添加一列“NM”，site为1时，NM为1，site为2时，NM为0，site为3时，NM为2，site为4时，NM为1，site为5时，NM为1，site为6时，NM为2
df['NM'] = df.site.map({1: 1, 2: 0, 3: 2, 4: 1, 5: 1, 6: 2})

#在df后添加一列"NO",site为1时，NO为0，site为2时，NO为1，site为3时，NO为0，site为4时，NO为1，site为5时，NO为2，site为6时，NO为1
df['NO'] = df.site.map({1: 0, 2: 1, 3: 0, 4: 1, 5: 2, 6: 1})

#删除df中包含NaN的行
df = df.dropna()
df.to_csv('test.csv', index=False)



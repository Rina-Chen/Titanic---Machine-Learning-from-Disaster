import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#1.读取数据集
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#数据可视化
#2.查看缺失值
# print(train.info())
# print(test.info())

# sex & survived 关系
sns.boxplot(x='Sex', y='Survived', data=train)
plt.title('Sex & Survived')
plt.show()

# 年龄与生存率的分布
sns.histplot(train['Age'].dropna(),kde=True,bins=30)
plt.title('Age & Survived')
plt.show()

#3.数据预处理（补缺失值）
train['Age'].fillna(train['Age'].median(),inplace=True)
test['Age'].fillna(test['Age'].median(),inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)

test['Fare'].fillna(test['Fare'].median(),inplace=True)

# 填充Cabin的缺失值为 'Unknown'
train['Cabin'].fillna('Unknown', inplace=True)
# 提取Cabin的甲板号，将NaN视为 'U'（代表Unknown）
train['Deck'] = train['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')

#4.将分类变量转化为数值
label = LabelEncoder()
train['Sex'] = label.fit_transform(train['Sex'])
test['Sex'] = label.fit_transform(test['Sex'])
train['Embarked'] = label.fit_transform(train['Embarked'])
test['Embarked'] = label.fit_transform(test['Embarked'])

#5.特征选择
features = ['Pclass','Sex','Age','Fare','Embarked']

#.6训练数据
x_train = train[features]
y_train = train['Survived']

#7.测试数据
x_test = test[features]

#8.模型训练（RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)

#9.预测准确值
train_predictions = model.predict(x_train)
accuracy_score = accuracy_score(y_train,train_predictions)
print(f"Training accuracy:{accuracy_score:.2f}")

#10.预测测试集
test_preditions = model.predict(x_test)

#11.提交
submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived':test_preditions})

submission.to_csv('submission.csv',index=False)
from sklearn.linear_model import Ridge, LinearRegression, Lasso 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.datasets import fetch_california_housing as fch # 加利福尼亚房屋价值数据集
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

housevalue=fch()#导入数据集
X=pd.DataFrame(housevalue.data)
X.columns=housevalue.feature_names
print(X.head())
vif=[variance_inflation_factor(X.values,X.columns.get_loc(i)) for i in X.columns]
print('数据的VIF为:',vif)#通常情况下，当VIF<10,说明不存在多重共线性；当10<=VIF<100,存在较强的多重共线性，当VIF>=100,存在严重多重共线性
y=housevalue.target
print('房价的最大值为',y.max(),'\n房价的最小值为:',y.min())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
# 重置特征矩阵的索引
for i in [X_train,X_test]:
    i.index=range(i.shape[0])

linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_hat=linreg.predict(X_test)
MSE_lin=MSE(y_hat,y_test)
print('多元线性回归模型的MSE为:',MSE_lin)
r2_lin=linreg.score(X_test,y_test)
n,p=X_test.shape
r2_lin_adj=1-((1-r2_lin)*(n-1))/(n-p-1)
print('多元线性回归模型的adjustR2为:',r2_lin_adj)

ridge=Ridge(alpha=0.5)
ridge.fit(X_train,y_train)
y_hat1=ridge.predict(X_test)
MSE_ridge=MSE(y_hat1,y_test)
print('岭回归模型的MSE为:',MSE_ridge)
r2_ridge=ridge.score(X_test,y_test)
n,p=X_test.shape
r2_ridge_adj=1-((1-r2_ridge)*(n-1))/(n-p-1)
print('岭回归模型的adjustR2为:',r2_ridge_adj)

lasso=Lasso(alpha=0.0020729217795953697)
lasso.fit(X_train,y_train)
y_hat2=lasso.predict(X_test)
MSE_lasso=MSE(y_hat2,y_test)
print('Lasso回归模型的MSE为:',MSE_lasso)
r2_lasso=lasso.score(X_test,y_test)
n,p=X_test.shape
r2_lasso_adj=1-((1-r2_lasso)*(n-1))/(n-p-1)
print('Lasso模型的R2为:',r2_lasso_adj)

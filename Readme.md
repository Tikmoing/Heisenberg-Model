# 一维海森堡模型

![1.png](https://raw.githubusercontent.com/Tikmoing/Heisenberg-Model/main/png/1.png)

## Heisenberg_Limit.py
该文件用来计算L=4，6，8，10，12情况下，海森堡模型的最低两个能级，并求出能隙。
绘制能隙和1/L的的关系图，发现可以利用一阶和二阶进行拟合，得到L趋向于无穷大时的能量。
| L    | 1/L         | E0           | E1           | d E         |
| ---- | ----------- | ------------ | ------------ | ----------- |
| 4    | 0.250000000 | -2.000000000 | -1.000000000 | 1.000000000 |
| 6    | 0.166666667 | -2.802775638 | -2.118033989 | 0.684741649 |
| 8    | 0.125000000 | -3.651093409 | -3.128419064 | 0.522674345 |
| 10   | 0.100000000 | -4.515446354 | -4.092207347 | 0.423239007 |
| 12   | 0.083333333 | -5.387390917 | -5.031543404 | 0.355847513 |

![2](https://raw.githubusercontent.com/Tikmoing/Heisenberg-Model/main/png/2.png)

## Heisenberg_tempature.py

根据配分函数求温度和比热的关系

![3](https://raw.githubusercontent.com/Tikmoing/Heisenberg-Model/main/png/3.png)

import random
import numpy as np
from sklearn.datasets import load_digits
import dann
import rf
import plotly.graph_objects as go

digits = load_digits()
X = digits.data
target = digits.target

ddd = dann.dann()
X = ddd.normaniz(X)
# target = ddd.one_hot_encoding(target)

X_train, X_val, X_test, t_train, t_val, t_test = ddd.peremeshat_razd(X, target)

K = 10
D = 64

top = []
random.seed()
for i in range(30):
    # L_1, L_2, M = np.random.randint(10, 40), np.random.randint(5, 35), np.random.randint(5, 20)
    L_1 = np.random.randint(5, 20)
    L_2 = np.random.randint(5, 15)
    M = np.random.randint(3, 12)

    d = rf.DT(3, 0.05, 25, K, M, L_1, L_2)
    print(str(i)+"   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    root = [rf.Node() for l in range(M)]
    for r in root:
        d.build_tree(X_train, t_train, r, 0)
    err = d.accuracy_random_forest(t_val, X_val, root)
    top.append([err, M, L_1, L_2, d.accuracy_random_forest(t_test, X_test, root)])
    if len(top) > 10:
        top.sort()
        top.pop()


fig = go.Figure()
fig.add_trace(go.Scatter(x=[i for i in range(10)], y=[i[0] for i in top], mode='markers', name="t"))
c_M = np.array([top[i][1] for i in range(10)])
c_L_1 = np.array([top[i][2] for i in range(10)])
c_L_2 = np.array([top[i][3] for i in range(10)])
c_acc_val = np.array([top[i][4] for i in range(10)])
c_acc_test = np.array([top[i][0] for i in range(10)])
customdata = np.stack((c_M, c_L_1, c_L_2,c_acc_val,c_acc_test), axis=-1)
fig.update_traces(customdata=customdata, hovertemplate="M=%{customdata[0]}<br>L_1=%{customdata[1]}<br>"
                                                       "L_2=%{customdata[2]}<br>accuracy_val=%{customdata[3]}<br>"
                                                       "accuracy_test=%{customdata[4]}"+'<extra></extra>')
fig.update_layout(xaxis_title="function",
                  yaxis_title="error_valid",)
fig.show()
fig.write_html("top.html")




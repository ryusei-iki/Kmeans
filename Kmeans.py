import numpy as np
import matplotlib.pyplot as plt
import math

#データ生成
#データパラメータの設定
x_num=400
class_num=4
x_dim=2
sigma=[[1,0],[0,1]]
average_list=[[2,2],[-2,2],[-2,-2],[2,-2]]
#教師データの生成
x=np.zeros((class_num,x_num//class_num,x_dim))
for i in range(class_num):
    x[i]=np.random.multivariate_normal(average_list[i],sigma,x_num//class_num)
x=x.reshape(-1,x_dim)

#ラベルデータの生成
y=[math.floor(i/x_num*class_num) for i in range(x_num)]
real_y=[math.floor(i/x_num*class_num) for i in range(x_num)]

#描画のための色の配列の作成
colors=np.array(['r','g','b','c'])

#代表点の作成
cluster_ave=np.random.uniform(-4,4,(class_num,2))


#入力データの描画
fig=plt.figure()
gs=fig.add_gridspec(1,1)
ax1=fig.add_subplot(gs[0,0],aspect='equal')
ax1.set_title('Ground truth')
ax1.scatter(x[:,0],x[:,1],c=colors[real_y])
plt.show()

#描画の準備
fig=plt.figure(figsize=(17,9))
gs=fig.add_gridspec(1,2)
fig.suptitle('Kmeans')
ax1=fig.add_subplot(gs[0,0],aspect='equal')
ax2=fig.add_subplot(gs[0,1],aspect='equal')
#学習スタート
for epoch in range(100):
    con=[0 for i in range(class_num)]
    for i in range(x_num):
        d_min=float('inf')
        for j in range(class_num):
            if((x[i][0]-cluster_ave[j][0])**2+(x[i][1]-cluster_ave[j][1])**2<d_min):
                y[i]=j
                d_min=(x[i][0]-cluster_ave[j][0])**2+(x[i][1]-cluster_ave[j][1])**2
        con[y[i]]=con[y[i]]+1
    plt.suptitle(epoch)
    ax1.set_title('{}:epoch'.format(epoch))
    ax1.scatter(x[:,0],x[:,1],c=colors[y])
    ax1.scatter(cluster_ave[:,0],cluster_ave[:,1],c=['r','g','b','c'],s=500)
    ax2.set_title('Ground Truth')
    ax2.scatter(x[:,0],x[:,1],c=colors[real_y])
    plt.pause(0.1)
    ax1.cla()
    ax2.cla()

    wa=[[0,0]for i in range(class_num)]
    for i in range(x_num):
        wa[y[i]][0]=x[i][0]+wa[y[i]][0]
        wa[y[i]][1]=x[i][1]+wa[y[i]][1]
    check=0
    for i in range(class_num):
        for j in range(x_dim):
            if(cluster_ave[i][j]==wa[i][j]/con[i]):
                check=check+1
            cluster_ave[i][j]=wa[i][j]/con[i]
    if(check==class_num*x_dim):
        break
ax1.set_title('{}:epoch'.format(epoch))
ax1.scatter(x[:,0],x[:,1],c=colors[y])
ax1.scatter(cluster_ave[:,0],cluster_ave[:,1],c=['r','g','b','c'],s=500)
ax2.set_title('Ground Truth')
ax2.scatter(x[:,0],x[:,1],c=colors[real_y])
plt.show()
print('end')
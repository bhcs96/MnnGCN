# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:55:36 2019

@author: LIU-CY
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:10:13 2019

@author: LIU-CY
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:06:18 2019

@author: LIU-CY
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse as sp
#tfd = tf.contrib.distributions
from nvae import BSVGAE
from utils import preprocess_graph, sparse_to_tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


#rtn_data = pd.read_csv('../../Dataset/rtn.var.seurat.csv',index_col=0)
rtn_count = pd.read_csv('../../Dataset/rcc/prcc_rcc1_rcc2_count_varfea1k.csv',index_col=0)
rtn_label = pd.read_csv('../../Dataset/rcc/prcc_rcc1_rcc2_celltype.csv', index_col=0)
rtn_source = pd.read_csv("../../Dataset/rcc/prcc_rcc1_rcc2_batch.csv", index_col=0)
rtn_adj = pd.read_csv("../../Dataset/rcc/prcc_rcc1_rcc2_adj_sparse.csv", index_col=0)
#rtn_adj = pd.read_csv("../../Dataset/retina/retina_mnnadj(3773x3650)_2nd_sparse.csv", index_col=0)

label = rtn_label.iloc[:, 0]
source = np.asarray(rtn_source.iloc[:, 0])
source_set = list(set(source))
source_dict = {source_set[i]:i  for i in range(len(source_set))}
source_id = [source_dict[s] for s in source]
#fun_label[fun_label=="CD14+ Monocytes"] = "Monocyte"
#fun_label[fun_label=="FCGR3A+ Monocytes"] = "Monocyte"

# input counts
count = np.asarray(rtn_count).astype(np.int32)
expr = count.T
#count_log = np.log(count.T + 1)
#expr_l1 = np.sum(count_log, axis=1)
#expr = count_log/expr_l1[:,None]
#expr = sparse_to_tuple(sp.coo_matrix(expr))

#expr = np.asarray(rtn_count).astype(np.int32)
#expr = expr.T

in_dim = expr.shape[1]
in_num = expr.shape[0]
#in_edge = rtn_adj.shape[0]+in_num

# prepare adj
adj = sp.coo_matrix((rtn_adj.iloc[:, 3], (rtn_adj.iloc[:, 0], rtn_adj.iloc[:, 1])), shape=(in_num, in_num))
adj = adj + sp.eye(adj.shape[0])
#adj = sp.coo_matrix(rtn_adj)
adj_norm = preprocess_graph(adj)
#adj_norm_de = preprocess_graph_fordecoder(adj)

adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)

norm = in_num * in_num / float((in_num * in_num - adj.toarray().sum()) * 2)
pos_weight = float(in_num * in_num - adj.toarray().sum()) / adj.toarray().sum()

# initiation
batch_remove = False
var = True
n_batch = 2
n_latent = 16
x_dist = "nb" #"mn"
epoch = 1200
n_node = in_num

# create object
bvae = BSVGAE(in_dim = in_dim, n_latent=n_latent, n_batch=n_batch, var=var, x_dist=x_dist, n_node = n_node, norm=norm, pos_weight=pos_weight, batch_remove=batch_remove)

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# iteration
record_loss = np.zeros(epoch)

for e in range(epoch):
    _, elbo, re_loss, kl_loss = sess.run([bvae.optimizer, bvae.elbo, bvae.re_loss, bvae.kl_loss], feed_dict = {bvae.x: expr, bvae.batch_id: source_id, bvae.adj: adj_norm, bvae.adj_orig: adj_label})
    record_loss[e] = elbo
    if not e % 50:
        elbo, re_loss, kl_loss = sess.run([bvae.elbo, bvae.re_loss, bvae.kl_loss], feed_dict = {bvae.x: expr, bvae.batch_id: source_id, bvae.adj: adj_norm, bvae.adj_orig: adj_label})
        print("Epoch: %d   Loss: %f"%(e, elbo))
res = sess.run(bvae.z_mu, feed_dict = {bvae.x: expr, bvae.batch_id: source_id, bvae.adj: adj_norm, bvae.adj_orig: adj_label})


### LOSS
plt.plot(range(epoch), record_loss)
plt.xlabel("Epoch")
plt.ylabel("Elbo")
plt.title("LOSS") 
plt.savefig("./Result/rtn_bvae_loss.pdf")

### tsne
pca = PCA(n_components=2)
hidden_pca = pca.fit_transform(res)

tsne=TSNE(perplexity=40, n_iter=1000)
tsne.fit_transform(res)
hidden_tsne=tsne.embedding_

hidden_umap = umap.UMAP().fit_transform(res)


### Visualization of sig_cell
### z0 z1
label_set = np.sort(list(set(label)))
n_label = len(label_set)
figure1=plt.figure() 
ax=figure1.add_subplot(111)
colo = ['r','b','g','y','pink','skyblue','gold','greenyellow','brown','orange','purple','cyan','black','magenta','lightgreen','olive']
for i in range(n_label):
    zp = res[np.asarray(label)==label_set[i], 0:2]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(label_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=8,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure1.savefig("./Result/rtn_bvae_celltype_vgae01.pdf")


### pca
label_set = np.sort(list(set(label)))
n_label = len(label_set)
figure1=plt.figure() 
ax=figure1.add_subplot(111)
for i in range(n_label):
    zp = hidden_pca[np.asarray(label)==label_set[i], :]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(label_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=8,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure1.savefig("./Result/rtn_bvae_celltype_pca.pdf")

### tsne
label_set = np.sort(list(set(label)))
n_label = len(label_set)
figure1=plt.figure() 
ax=figure1.add_subplot(111)
for i in range(n_label):
    zp = hidden_tsne[np.asarray(label)==label_set[i], :]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(label_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=8,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure1.savefig("./Result/rtn_bvae_celltype_tsne.pdf")


### tsne
label_set = np.sort(list(set(label)))
n_label = len(label_set)
figure1=plt.figure() 
ax=figure1.add_subplot(111)
for i in range(n_label):
    zp = hidden_umap[np.asarray(label)==label_set[i], :]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(label_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=8,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure1.savefig("./Result/rtn_bvae_celltype_tsne.pdf")


### Visualization of sig_cell
source_set = np.sort(list(set(source)))
n_source = len(source_set)
figure2=plt.figure() 
ax=figure2.add_subplot(111)
for i in range(n_source):
    zp = hidden_tsne[np.asarray(source)==source_set[i],:]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(source_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=8,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure2.savefig("./Result/rtn_bvae_source_tsne.pdf")

### Visualization of sig_cell
source_set = np.sort(list(set(source)))
n_source = len(source_set)
figure2=plt.figure() 
ax=figure2.add_subplot(111)
for i in range(n_source):
    zp = hidden_umap[np.asarray(source)==source_set[i],:]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(source_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=8,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure2.savefig("./Result/rtn_bvae_source_tsne.pdf")

out_res = pd.DataFrame(res)
out_res.to_csv("./Dataset/vae_sc_normal_res.csv")
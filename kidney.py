# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:31:52 2019

@author: LIU-CY
"""



import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.sparse as sp
#tfd = tf.contrib.distributions
from nvae import BSVGAE
from utils import preprocess_graph, preprocess_graph1, sparse_to_tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


#rcc_data = pd.read_csv('../../Dataset/rcc.var.seurat.csv',index_col=0)
rcc_count = pd.read_csv('../../Dataset/kidney/kidney_var_count.csv',index_col=0)
rcc_label = pd.read_csv('../../Dataset/kidney/kidney_label.csv', index_col=0)
rcc_source = pd.read_csv("../../Dataset/kidney/kidney_source.csv", index_col=0)
rcc_adj = pd.read_csv("../../Dataset/kidney/kidney_adj_sparse.csv", index_col=0)

label = rcc_label.iloc[:, 0]
source = np.asarray(rcc_source.iloc[:, 0])
source_set = list(set(source))
source_dict = {source_set[i]:i  for i in range(len(source_set))}
source_id = [source_dict[s] for s in source]
#fun_label[fun_label=="CD14+ Monocytes"] = "Monocyte"
#fun_label[fun_label=="FCGR3A+ Monocytes"] = "Monocyte"

# input counts
expr = np.asarray(rcc_count).astype(np.int32)
expr = expr.T
#count = np.asarray(rcc_count).astype(np.int32)
#count_log = np.log(count.T + 1)
#expr_l2 = np.linalg.norm(count_log, axis=1)
#expr = count_log/expr_l2[:,None]
#expr = sparse_to_tuple(sp.coo_matrix(expr))

in_dim = rcc_count.shape[0]
in_num = rcc_count.shape[1]
in_edge = rcc_adj.shape[0]+in_num

# prepare adj
adj = sp.coo_matrix((rcc_adj.iloc[:, 2], (rcc_adj.iloc[:, 0], rcc_adj.iloc[:, 1])), shape=(in_num, in_num))
adj = adj + sp.eye(adj.shape[0])

adj_norm = preprocess_graph(adj)

adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)

norm = in_num * in_num / float((in_num * in_num - in_edge) * 2)
pos_weight = float(in_num * in_num - in_edge) / in_edge

# initiation
batch_remove = False
var = True
n_batch = 2
n_latent = 32
x_dist = "nb" #"mn"
epoch = 300
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
#plt.savefig("./Result/rcc_bvae_loss.pdf")

### tsne
pca = PCA(n_components=2)
hidden_pca = pca.fit_transform(res)

tsne=TSNE(perplexity=20, n_iter=1000)
tsne.fit_transform(res)
hidden_tsne=tsne.embedding_


### Visualization of sig_cell
### z0 z1
label_set = np.sort(list(set(label)))
n_label = len(label_set)
figure1=plt.figure() 
ax=figure1.add_subplot(111)
colo = ['r','b','g','y','pink','skyblue','gold','greenyellow','brown']
for i in range(n_label):
    zp = res[np.asarray(label)==label_set[i], 0:2]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(label_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=4,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure1.savefig("./Result/rcc_bvae_celltype_vgae01.pdf")


### pca
label_set = np.sort(list(set(label)))
n_label = len(label_set)
figure1=plt.figure() 
ax=figure1.add_subplot(111)
colo = ['r','b','g','y','pink','skyblue','gold','greenyellow','brown']
for i in range(n_label):
    zp = hidden_pca[np.asarray(label)==label_set[i], :]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(label_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=4,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure1.savefig("./Result/rcc_bvae_celltype_pca.pdf")

### tsne
label_set = np.sort(list(set(label)))
n_label = len(label_set)
figure1=plt.figure() 
ax=figure1.add_subplot(111)
colo = ['r','b','g','y','pink','skyblue','gold','greenyellow','brown']
for i in range(n_label):
    zp = hidden_tsne[np.asarray(label)==label_set[i], :]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(label_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=4,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure1.savefig("./Result/rcc_bvae_celltype_tsne.pdf")


### Visualization of sig_cell
source_set = np.sort(list(set(source)))
n_source = len(source_set)
figure2=plt.figure() 
ax=figure2.add_subplot(111)
colo = ['r','b','g','y','pink','skyblue','gold','greenyellow','brown']
for i in range(n_source):
    zp = hidden_tsne[np.asarray(source)==source_set[i],:]
    ax.scatter(zp[:,0],zp[:,1], c=colo[i], s = 2) 
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

ax.legend(source_set,scatterpoints=1,loc='upper center',
          bbox_to_anchor=(0.5,-0.08),ncol=4,
          fancybox=True,
          prop={'size':8}
          )
ax.set_xlabel(xlabel="VAE_1")
ax.set_ylabel(ylabel="VAE_2")
ax.set_title('Normal VAE',fontsize=12)
plt.show()
figure2.savefig("./Result/rcc_bvae_source_tsne.pdf")


out_res = pd.DataFrame(res)
out_res.to_csv("./Dataset/vae_sc_normal_res.csv")
# Memo
from sklearn.cluster import KMeans
import numpy as np
from math import sqrt
import math
import glog as log

def intersect(a, b):
    return list(set(a) & set(b))

def union(a, b):
    return list(set(a) | set(b))


# The target vertex
#vertex_idx0 = int(1)
#vertex_idx1 = int(21)

graph_raw = np.loadtxt('./Graph.txt', dtype='int')
node_list = graph_raw.flatten().tolist()
node_list = list(set(node_list))
node_list = sorted(node_list, key=int)  
print node_list
#print node_list,len(node_list)
#print graph_raw,len(graph_raw)

N = len(node_list)
simM = np.zeros((N,N))
raw = np.zeros((N,N))
e = np.zeros((N,N))
A = np.zeros((N,N))
a = 0.1
# check if the given vertex idx is available
#list_graph_raw_c0 = graph_raw[:,0].tolist()
#list_graph_raw_c1 = graph_raw[:,1].tolist()

for i in range(N):
    log.info('%d',i)
    for j in range(N):
        vertex_idx0 = node_list[i]
        vertex_idx1 = node_list[j]
        #print vertex_idx0, vertex_idx1
        #assert (vertex_idx0 in list_graph_raw_c0) or (vertex_idx0 in list_graph_raw_c1)
        #assert (vertex_idx1 in list_graph_raw_c0) or (vertex_idx1 in list_graph_raw_c1)
        
        idx0_check_c0 = np.where(graph_raw[:,0]==vertex_idx0)
        idx0_check_c1 = np.where(graph_raw[:,1]==vertex_idx0)
        idx1_check_c0 = np.where(graph_raw[:,0]==vertex_idx1)
        idx1_check_c1 = np.where(graph_raw[:,1]==vertex_idx1)
        #print idx0_check_c0,idx0_check_c1
        #print idx1_check_c0,idx1_check_c1
        set_idx0 = graph_raw[idx0_check_c0,1].tolist()[0]
        set_idx0.extend(graph_raw[idx0_check_c1,0].tolist()[0])
        set_idx1 = graph_raw[idx1_check_c0,1].tolist()[0]
        set_idx1.extend(graph_raw[idx1_check_c1,0].tolist()[0])
        if (vertex_idx0 in set_idx1) or (vertex_idx1 in set_idx0):
            raw[i][j] = 1
            raw[j][i] = 1
            #print 'NN check **',vertex_idx0,vertex_idx1,set_idx0,set_idx1
        
        # remove the duplicated elements if necessary, here
        #print set_idx0
        #print set_idx1
        # union and intersection of set_idx0 & set_idx1
        sim_denominator = len(union(set_idx0, set_idx1))
        sim_numerator = len(intersect(set_idx0, set_idx1))
        #print sim_numerator,sim_denominator
        similarity = sim_numerator / float(sim_denominator)
        simM[i][j] = similarity #[][]?
        simM[j][i] = similarity
        #if similarity != 0:
        #    print vertex_idx0,vertex_idx1,set_idx0,set_idx1,similarity
            

print simM
print 'Vertex similarity calculation is done!'

#PPR
for i in range(N):
    for j in range(N):
        if raw[i][j]==1:
            e[i][j]=1/sum(raw[j])
            A[i][j]=1/sum(raw[i])

p_ini = e
threshold = 10e-5
dis = float('inf')
while dis > threshold:
    p = (1-a)*np.dot(A,p_ini)+a*e
    p_ini = p
    dis = sum(sum((p-p_ini)**2))

print 'PPR similarity calculation is done!'

kmeans_Ver_Sim = KMeans(n_clusters=2).fit(simM)
print 'Vertex similarity K-means is done!'
kmeans_PPR = KMeans(n_clusters=2).fit(p)
print 'PPR similarity K-means is done!'
print kmeans_Ver_Sim.labels_
print kmeans_PPR.labels_
# graph_sort_c0 = np.argsort(graph_raw, axis=0)
# graph_sort_c1 = np.argsort(graph_raw, axis=1)


# purity
results = {}
label_raw = np.loadtxt('./Labels.txt', dtype='int')
for i in range(len(label_raw)):
    node = label_raw[i,0]
    label = label_raw[i,1]
    results[node] = label
for index, kmeans_label in enumerate(kmeans_Ver_Sim.labels_):
    if node_list[index] not in results:
        continue
    #print index, kmeans_label,node_list[index]
    print index,node_list[index], kmeans_label, results[node_list[index]]
print 'Vertex results!'
for index, kmeans_label in enumerate(kmeans_PPR.labels_):
    if node_list[index] not in results:
        continue
    #print index, kmeans_label,node_list[index]
    print index,node_list[index], kmeans_label, results[node_list[index]]
print 'PPR K-means results!'
class_one = {"Clinton": 0, "Trump": 0}
class_two = {"Clinton": 0, "Trump": 0}
number = 0
for index, kmeans_label in enumerate(kmeans_Ver_Sim.labels_):
#for index, kmeans_label in enumerate(kmeans_PPR.labels_):
    number += 1
    index = node_list[index]
    if index not in results:
        continue
    if kmeans_label == 0:
        if results[index] == 0:
            class_one["Clinton"] += 1
        else:
            class_one["Trump"] += 1
    else:
        if results[index] == 0:
            class_two["Clinton"] += 1
        else:
            class_two["Trump"] += 1

if class_one["Clinton"] < class_one["Trump"]:
    label_class_one = "Trump"
else:
    label_class_one = "Clinton"

if class_two["Clinton"] < class_two["Trump"]:
    label_class_two = "Trump"
else:
    label_class_two = "Clinton"
number = float(number)
purity = (class_one[label_class_one] + class_two[label_class_two]) / number
print "purity: ",purity

#entropy
print class_one
number_class_one = 0
for i, j in class_one.items():
    number_class_one += j
print class_two
number_class_two = 0
for i, j in class_two.items():
    number_class_two += j

entropy = (-1)*((number_class_one*math.log(number_class_one/number)) + (number_class_two*math.log(number_class_two/number)))
print "entropy: ",entropy

#NMI
clinton = class_one["Clinton"] + class_two["Clinton"]
trump = class_one["Trump"] + class_two["Trump"]
H = -clinton*math.log(clinton/number) + (-trump*math.log(trump/number))
I = -class_one["Clinton"]*math.log(class_one["Clinton"]/number) \
    + (-class_one["Trump"]*math.log(class_one["Trump"]/number)) \
    + (-class_two["Clinton"]*math.log(class_two["Clinton"]/number)) \
    + (-class_two["Trump"]*math.log(class_two["Trump"]/number))
NMI = I / ((entropy + H) / 2)
print "NMI: ",NMI

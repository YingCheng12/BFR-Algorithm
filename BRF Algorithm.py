from sklearn.cluster import KMeans
import numpy as np
import time
import sys
import math

## build_dict function is to create a dict, key = label, value = [index]
def build_dict_index(all_labels):
    dict_label_indexofdata = {}  ##{label:[index1, index2]}
    for indexoflabel in range(len(all_labels)):
        if all_labels[indexoflabel] in dict_label_indexofdata:
            dict_label_indexofdata[all_labels[indexoflabel]].append(indexoflabel)
        else:
            dict_label_indexofdata[all_labels[indexoflabel]] = []
            dict_label_indexofdata[all_labels[indexoflabel]].append(indexoflabel)
    return dict_label_indexofdata


def build_dict_data(all_labels, data):
    dict_label_data = {}
    for indexoflabel in range(len(all_labels)):
        # print(indexoflabel)
        temp_label = all_labels[indexoflabel]
        if temp_label in dict_label_data:
            dict_label_data[temp_label].append(data[indexoflabel])
        else:
            dict_label_data[temp_label] = []
            dict_label_data[temp_label].append(data[indexoflabel])
    return dict_label_data


def find_index_RS(dict_label_indexofdata):
    RS_index = []
    for i in dict_label_indexofdata:
        if len(dict_label_indexofdata[i]) == 1:
            temp_index = dict_label_indexofdata[i][0]
            RS_index.append(temp_index)
    return RS_index

# def find_index_notRS(dict_label_indexofdata):
#     notRS_index = []
#     for i in dict_label_indexofdata:
#         if len(dict_label_indexofdata[i]) != 1:
#             temp_index = dict_label_indexofdata[i][0]
#             RS_index.append(temp_index)
#     return RS_index

def find_cs_data(dict_lable_data_s6):
    dict_lable_csdata = {}
    for key in dict_lable_data_s6:
        if len(dict_lable_data_s6[key]) != 1:
            dict_lable_csdata[key] = dict_lable_data_s6[key]
    return dict_lable_csdata


def generate_sum_count_sumsqr(dict_label_data):
    DS = []
    for key in dict_label_data:
        dict_label_data[key] = np.array(dict_label_data[key])
        # print(dict_label_data[key])
        temp_sum = np.sum(dict_label_data[key][:, 2:], axis=0)
        # print(temp_sum)

        # print(temp_sum)
        temp_count = len(dict_label_data[key])
        # print(temp_count)
        temp_sum_sq = np.sum([np.square(num) for num in dict_label_data[key][:, 2:]], axis=0)
        # print(temp_sum_sq)

        temp_index = []
        temp_label = []
        for i in dict_label_data[key]:
            # print(i)
            temp_index.append(i[0])
            temp_label.append(i[1])

        temp = [temp_sum, temp_count, temp_sum_sq, temp_index, temp_label]
        # print(temp)
        DS.append(temp)
    return np.array(DS)


def calculate_mahalanobis(datapoint,clusterdata):
    centroid = clusterdata[0] / clusterdata[1]
    sigma2 = clusterdata[2]/clusterdata[1] - centroid**2
    normalize = (datapoint - centroid)**2/sigma2
    # print(sigma2)
    distance = pow(normalize.sum(), 0.5)
    # distance = math.sqrt(np.sum(normalize))
    return distance




s = time.perf_counter()
file_path = sys.argv[1]
# file_path ="/Users/irischeng/INF553/Assignment/hw5/hw5_clustering.txt"
file = open(file_path, 'r')
file_list = []
for each_line in file.readlines():
    temp_line = each_line.strip("\n").split(",")
    temp_line_int = []
    # for each_feature in temp_line[2:]:
    for each_feature in temp_line:
        temp_line_int.append(float(each_feature))
    # print(temp_line_int)
    file_list.append(temp_line_int)


## file_list has 12 col, the first col is index, the second col is label
file_length = len(file_list)
# print(file_length)
percentage = 0.2
d=10
k = int(sys.argv[2])
# k = 10
threshold = 2 * pow(d,0.5)
pro = 0.8
i = 0

## init_data  is the first 20% data, type is np.array
init_data = np.array(file_list[:int(file_length*percentage)])
# print(init_data)
# print(len(init_data))
# print(init_data[:, 2:])

## step 2 k-means with a large k
kmeans_init = KMeans(n_clusters=10*k, random_state=i)
kmeans_init.fit(init_data[:, 2:])
dict_lable_indexofdata = build_dict_index(kmeans_init.labels_)


## step 3 move point to rs
RS_index = find_index_RS(dict_lable_indexofdata)   ## the index is the index of the orginal data, init_data here
# print(RS_index)
RS = init_data[RS_index]
# print(len(RS))

## step 4, run kmeans with k==10 in the data without the rs
init_data_remove = np.delete(init_data, RS_index, 0)
kmeans_equal_10 = KMeans(n_clusters=k,random_state=i)
kmeans_equal_10.fit(init_data_remove[:, 2:])

## step 5, use the result from 4 to generate ds
dict_lable_data = build_dict_data(kmeans_equal_10.labels_, init_data_remove)
# print(dict_lable_data)
DS = generate_sum_count_sumsqr(dict_lable_data)
# print(DS)
# print(type(DS[9][4]))
# print(DS[1])
# print(DS[:,1].)
number_ds_data = sum(DS[:, 1])
# print(number_ds_data)


## step 6, run k-means  in rs with large k to generate cs and rs
kmeans_s6 = KMeans(n_clusters= int(pro*len(RS)), random_state=i)
kmeans_s6.fit(RS[:, 2:])
dict_lable_indexofdata_s6 = build_dict_index(kmeans_s6.labels_)
# RS = []   ## empty the rs to store new rs
RS_index_s6 = find_index_RS(dict_lable_indexofdata_s6)
RS_new = RS[RS_index_s6]
# print(len(RS_new))
dict_lable_data_s6 = build_dict_data(kmeans_s6.labels_, RS)
cs_data = find_cs_data(dict_lable_data_s6)
# print(len(cs_data))

# print(cs_data)
if len(cs_data.keys())==0:
    number_cs_cluster = 0
    number_cs_data = 0
    number_rs_data = len(RS_new)
    CS = np.array([])
    # print(number_rs_data)
else:
    CS = generate_sum_count_sumsqr(cs_data)
    number_cs_cluster = len(CS)
    number_cs_data = sum(CS[:, 1])
    # print(number_cs_cluster)
    # print(number_cs_data)
    number_rs_data = len(RS_new)
    # print(number_rs_data)


result = []
temp_result = [number_ds_data, number_cs_cluster, number_cs_data, number_rs_data]
result.append(temp_result)
# print(temp_result)

## step 7, load another 20% of the data randomly
start = int(file_length*percentage)
end = start + int(file_length*percentage)

iteration = 1
while start< int(file_length*percentage)*1/percentage:
    if start != (1 / percentage - 1) * int(percentage * file_length):
        reload_data = np.array(file_list[start:end])

        for datapoint in reload_data:
            # print(datapoint)
            all_distance = []
            for i in range(len(DS)):
                temp_distance = calculate_mahalanobis(datapoint[2:], DS[i])
                all_distance.append(temp_distance)
            temp_min = min(all_distance)
            temp_index = all_distance.index(temp_min)
            # print(temp_index)
            # print()
            if temp_min < threshold:

                DS[temp_index][0] = DS[temp_index][0] + datapoint[2:]
                DS[temp_index][1] = DS[temp_index][1] + 1
                DS[temp_index][2] = DS[temp_index][2] + datapoint[2:] ** 2
                DS[temp_index][3] = np.append(DS[temp_index][3], datapoint[0])
                DS[temp_index][4] = np.append(DS[temp_index][4], datapoint[1])
                # print(DS[temp_index][3])
                # print(len(DS[temp_index][3]))

                # p_arr = np.append(p_arr, p_)
                # print(DS[temp_index])
            else:
                # if len(CS)!=0:
                all_distance_cs = []
                for j in range(len(CS)):
                    temp_distance = calculate_mahalanobis(datapoint[2:], CS[j])
                    all_distance_cs.append(temp_distance)
                temp_min_cs = min(all_distance_cs)
                temp_index_cs = all_distance_cs.index(temp_min_cs)
                # print(temp_index_cs)
                if temp_min_cs < threshold:
                    # print(temp_min_cs)
                    # print(all_distance_cs.index(temp_min_cs))
                    CS[temp_index_cs][0] = CS[temp_index_cs][0] + datapoint[2:]
                    CS[temp_index_cs][1] = CS[temp_index_cs][1] + 1
                    CS[temp_index_cs][2] = CS[temp_index_cs][2] + datapoint[2:] ** 2
                    CS[temp_index_cs][3] = np.append(CS[temp_index_cs][3], datapoint[0])
                    CS[temp_index_cs][4] = np.append(CS[temp_index_cs][4], datapoint[1])

                else:
                    RS_new = np.insert(RS_new, 0, datapoint, axis=0)

        temp_number_ds_data = sum(DS[:, 1])

        temp_RS = RS_new

        temp_iteration_kmeans = KMeans(n_clusters=int(pro * len(temp_RS)), random_state=i)
        temp_iteration_kmeans.fit(temp_RS[:, 2:])

        temp_dict_lable_indexofdata = build_dict_index(temp_iteration_kmeans.labels_)
        # # RS = []   ## empty the rs to store new rs
        temp_RS_index = find_index_RS(temp_dict_lable_indexofdata)
        temp_RS_new = temp_RS[temp_RS_index]
        temp_dict_lable_data = build_dict_data(temp_iteration_kmeans.labels_, temp_RS)
        temp_cs_data = find_cs_data(temp_dict_lable_data)
        temp_CS = generate_sum_count_sumsqr(temp_cs_data)

        temp_number_rs_data = len(temp_RS_new)

        for index_cs in range(len(temp_CS)):

            temp_centroid = temp_CS[index_cs][0] / temp_CS[index_cs][1]
            all_distance_merge = []
            for index in range(len(CS)):

                temp_distance = calculate_mahalanobis(temp_centroid, CS[index])
                all_distance_merge.append(temp_distance)
            temp_min_merge = min(all_distance_merge)
            temp_index_merge = all_distance_merge.index(temp_min_merge)

            if temp_min_merge < threshold:
                CS[temp_index_merge][0] = CS[temp_index_merge][0] + temp_CS[index_cs][0]
                # print(CS[index][0])
                CS[temp_index_merge][1] = CS[temp_index_merge][1] + temp_CS[index_cs][1]
                CS[temp_index_merge][2] = CS[temp_index_merge][2] + temp_CS[index_cs][2]
                CS[temp_index_merge][3] = np.append(CS[temp_index_merge][3], temp_CS[index_cs][3])
                CS[temp_index_merge][4] = np.append(CS[temp_index_merge][4], temp_CS[index_cs][4])

            else:
        #         # remain_cs = np.insert(remain_cs, 0, datapoint, axis = 0)
                CS = np.insert(CS, 0, temp_CS[index_cs], axis=0)

        temp_number_cs_data = sum(CS[:, 1])
        # print(temp_number_cs_data)
        temp_number_cs_cluster = len(CS)
        temp_result = [temp_number_ds_data, temp_number_cs_cluster, temp_number_cs_data, temp_number_rs_data]
        # print(temp_result)
        result.append(temp_result)

        RS_new = temp_RS_new

        start = end
        end = start + int(file_length * percentage)
        iteration = iteration + 1

    else:
        reload_data = np.array(file_list[start:])
        for datapoint in reload_data:
            # print(datapoint)
            all_distance = []
            for i in range(len(DS)):
                temp_distance = calculate_mahalanobis(datapoint[2:], DS[i])
                all_distance.append(temp_distance)
            temp_min = min(all_distance)
            temp_index = all_distance.index(temp_min)
            # print(temp_index)
            # print()
            if temp_min < threshold:

                DS[temp_index][0] = DS[temp_index][0] + datapoint[2:]
                DS[temp_index][1] = DS[temp_index][1] + 1
                DS[temp_index][2] = DS[temp_index][2] + datapoint[2:] ** 2
                DS[temp_index][3] = np.append(DS[temp_index][3], datapoint[0])
                DS[temp_index][4] = np.append(DS[temp_index][4], datapoint[1])
                # print(DS[temp_index][3])
                # print(len(DS[temp_index][3]))

                # p_arr = np.append(p_arr, p_)
                # print(DS[temp_index])
            else:
                # if len(CS)!=0:
                all_distance_cs = []
                for j in range(len(CS)):
                    temp_distance = calculate_mahalanobis(datapoint[2:], CS[j])
                    all_distance_cs.append(temp_distance)
                temp_min_cs = min(all_distance_cs)
                temp_index_cs = all_distance_cs.index(temp_min_cs)
                # print(temp_index_cs)
                if temp_min_cs < threshold:
                    # print(temp_min_cs)
                    # print(all_distance_cs.index(temp_min_cs))
                    CS[temp_index_cs][0] = CS[temp_index_cs][0] + datapoint[2:]
                    CS[temp_index_cs][1] = CS[temp_index_cs][1] + 1
                    CS[temp_index_cs][2] = CS[temp_index_cs][2] + datapoint[2:] ** 2
                    CS[temp_index_cs][3] = np.append(CS[temp_index_cs][3], datapoint[0])
                    CS[temp_index_cs][4] = np.append(CS[temp_index_cs][4], datapoint[1])

                else:
                    RS_new = np.insert(RS_new, 0, datapoint, axis=0)

        temp_number_ds_data = sum(DS[:, 1])

        temp_RS = RS_new

        temp_iteration_kmeans = KMeans(n_clusters=int(pro * len(temp_RS)), random_state=i)
        temp_iteration_kmeans.fit(temp_RS[:, 2:])

        temp_dict_lable_indexofdata = build_dict_index(temp_iteration_kmeans.labels_)
        # # RS = []   ## empty the rs to store new rs
        temp_RS_index = find_index_RS(temp_dict_lable_indexofdata)
        temp_RS_new = temp_RS[temp_RS_index]
        temp_dict_lable_data = build_dict_data(temp_iteration_kmeans.labels_, temp_RS)
        temp_cs_data = find_cs_data(temp_dict_lable_data)
        temp_CS = generate_sum_count_sumsqr(temp_cs_data)

        temp_number_rs_data = len(temp_RS_new)

        for index_cs in range(len(temp_CS)):

            temp_centroid = temp_CS[index_cs][0] / temp_CS[index_cs][1]
            all_distance_merge = []
            for index in range(len(CS)):

                temp_distance = calculate_mahalanobis(temp_centroid, CS[index])
                all_distance_merge.append(temp_distance)
            temp_min_merge = min(all_distance_merge)
            temp_index_merge = all_distance_merge.index(temp_min_merge)

            if temp_min_merge < threshold:
                CS[temp_index_merge][0] = CS[temp_index_merge][0] + temp_CS[index_cs][0]
                # print(CS[index][0])
                CS[temp_index_merge][1] = CS[temp_index_merge][1] + temp_CS[index_cs][1]
                CS[temp_index_merge][2] = CS[temp_index_merge][2] + temp_CS[index_cs][2]
                CS[temp_index_merge][3] = np.append(CS[temp_index_merge][3], temp_CS[index_cs][3])
                CS[temp_index_merge][4] = np.append(CS[temp_index_merge][4], temp_CS[index_cs][4])

            else:
        #         # remain_cs = np.insert(remain_cs, 0, datapoint, axis = 0)
                CS = np.insert(CS, 0, temp_CS[index_cs], axis=0)

        temp_number_cs_data = sum(CS[:, 1])
        # print(temp_number_cs_data)
        temp_number_cs_cluster = len(CS)
        temp_result = [temp_number_ds_data, temp_number_cs_cluster, temp_number_cs_data, temp_number_rs_data]
        # print(temp_result)
        result.append(temp_result)

        RS_new = temp_RS_new

        ## merge ds and cs
        cs_index = []
        for index_cs in range(len(CS)):
            # print(i)
            temp_centroid = CS[index_cs][0] / CS[index_cs][1]
            all_distance_final = []
            for index in range(len(DS)):
                # print(index)
                temp_distance = calculate_mahalanobis(temp_centroid, DS[index])
                all_distance_final.append(temp_distance)
                # print(temp_distance)
            temp_min_final = min(all_distance_final)
            temp_index_final = all_distance_final.index(temp_min_final)
            if temp_min_final < threshold:
                DS[temp_index_final][0] = DS[temp_index_final][0] + CS[index_cs][0]
                # print(CS[index][0])
                DS[temp_index_final][1] = DS[temp_index_final][1] + CS[index_cs][1]
                DS[temp_index_final][2] = DS[temp_index_final][2] + CS[index_cs][2]
                DS[temp_index_final][3] = np.append(DS[temp_index_final][3], CS[index_cs][3])
                DS[temp_index_final][4] = np.append(DS[temp_index_final][4], CS[index_cs][4])

            else:
                for each_index in list(CS[index_cs][3]):
                    # print(each_index)
                    cs_index.append(each_index)

        rs_index = RS_new[:, 0]
        # print(len(rs_index))
        # print(len(cs_index))
        all_rs_index = np.append(rs_index, np.array(cs_index))
        # print(len(all_rs_index))
        dict_index_predictLabel = {}
        for j in all_rs_index:
            dict_index_predictLabel[int(j)] = -1
        for i in range(k):
            for each_index in DS[i][3]:
                dict_index_predictLabel[int(each_index)] = i

        start = end
        end = start + int(file_length * percentage)
        iteration = iteration + 1




fileObject = open(sys.argv[3], 'w')
fileObject.write("The intermediate results:\n")
for i in range(len(result)):
    temp_round = i+1
    fileObject.write("Round"+" "+str(temp_round)+":")
    fileObject.write(str(result[i]).strip("[").strip(']')+"\n")
fileObject.write("\n")
fileObject.write("The clustering results:\n")
for i in sorted(dict_index_predictLabel.keys()):
    fileObject.write(str(int(i))+","+" ")
    fileObject.write(str(dict_index_predictLabel[i])+"\n")

fileObject.close()



# print(len(RS_update))
# print(final_DS)
# print(dict_index_predictLabel)
# print(result)

e = time.perf_counter()
print("duration:", e - s)

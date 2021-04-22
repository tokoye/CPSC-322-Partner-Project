"""
Charles Walker 
CPSC 322
Section 02
PA6
"""
import math 
import numpy as np
import importlib
import copy
import random
import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable
from tabulate import tabulate

# import mysklearn.myutils
# importlib.reload(mysklearn.myutils)
# import mysklearn.myutils as myutils

# # uncomment once you paste your mypytable.py into mysklearn package


# import mysklearn.myclassifiers
# importlib.reload(mysklearn.myclassifiers)
# from mysklearn.myclassifiers import MyKNeighborsClassifier, MySimpleLinearRegressor

# import mysklearn.myevaluation
# importlib.reload(mysklearn.myevaluation)
# import mysklearn.myevaluation as myevaluation


def get_column(table, header, col_name):
    col_index = header.index(col_name)
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def get_col_byindex(table, i):
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[i] != "NA":
            col.append(row[i])
    return col

def get_frequencies(table, header, col_name):
    col = get_column(table, header, col_name)

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def group_by(table, header, group_by_col_name):
    col = get_column(table, header, group_by_col_name)
    col_index = header.index(group_by_col_name)
    
    # we need the unique values for our group by column
    group_names = sorted(list(set(col))) # e.g. 74, 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], [], []]
    
    # algorithm: walk through each row and assign it to the appropriate
    # subtable based on its group_by_col_name value
    for row in table:
        group_by_value = row[col_index]
        # which subtable to put this row in?
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy()) # shallow copy
    
    return group_names, group_subtables

def compute_equal_width_cutoffs(values, num_bins):
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    # bin_width is likely a float
    # if your application allows for ints, use them
    # we will use floats
    # np.arange() is like the built in range() but for floats
    cutoffs = list(np.arange(min(values), max(values), bin_width)) 
    cutoffs.append(max(values))
    # optionally: might want to round
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 
    
def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs

def compute_slope_intercept(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    # y = mx + b => b = y - mx
    b = mean_y - m * mean_x
    return m, b 

def compute_euclidean_distance(v1, v2):
    print(v1, v2)
    assert len(v1) == len(v2)

    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist 

#pa3 add-ons
#
#

def conv_num(mypy):
    mypy.convert_to_numeric
    pass

def load_data(filename):
    mypytable = MyPyTable()
    mypytable.load_from_file(filename)
    return mypytable

def get_min_max(values):
    return min(values), max(values)

def binary_freq(mypy, col_name):
    mypy.convert_to_numeric()
    col = get_mypycol(mypy, col_name)
    freq = 0
    for i in range(len(col)):
        if col[i] == 1:
            freq += 1

    return col_name, freq

def percent_compare(mypy, col_names, total, get_sum=True):
    conv_num(mypy)
    percentages = []
    if get_sum == False:
        for i in range(len(col_names)):
            col = get_mypycol(mypy, col_names[i])
            col2 = []
            for j in range(len(col)):
                if col[j] != 0:
                    col2.append(col[j])
            col_total = len(col2)
            prcnt = col_total / total
            percentages.append(prcnt)
    if get_sum == True:
        for i in range(len(col_names)):
            col = get_mypycol(mypy, col_names[i])
            col_total = sum(col)
            prcnt = col_total / total
            percentages.append(prcnt)
    return col_names, percentages

# pa4 add-ons
#
#
def mpg_rating(val):
    rating = 0
    if val <=13:
        rating = 1
    elif val == 14:
        rating = 2
    elif 15 <= val < 17:
        rating = 3
    elif 17 <= val < 20:
        rating = 4
    elif 20 <= val < 24:
        rating = 5
    elif 24 <= val < 27:
        rating = 6
    elif 27 <= val < 31:
        rating = 7
    elif 31 <= val < 37:
        rating = 8
    elif 37 <= val < 45:
        rating = 9
    elif val >= 45:
        rating = 10
    return rating

def get_freq_str(col):
    
    header = ["y"]
    col_mypy = MyPyTable(header, col)

    dups = col_mypy.ordered_col(header)
    values = []
    counts = []

    for value in dups:
        if value not in values:
            # first time we have seen this value
            values.append(str(value))
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def get_accuracy(actual, pred):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            correct+=1
    return correct/len(actual)

def get_mypycol(mypy, col_name):
    return mypy.get_column(col_name, False)

def get_freq_1col(col):
    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def get_rand_rows(table, num_rows):
    rand_rows = []
    for i in range(num_rows):
        rand_rows.append(table.data[random.randint(0,len(table.data))-1])
    return rand_rows

def categorize_weight(val):
    if val < 2000:
        weight = 1
    elif val < 2500:
        weight = 2
    elif val < 3000:
        weight = 3
    elif val < 3500:
        weight = 4
    else:
        weight = 5
    return weight

def convert_weights(weight):
        res = []
        for val in weight:
            res.append(categorize_weight(val))
        return res

def print_results(rows, actual, predicted):
        for i in range(len(rows)):
            print('instance:', rows[i])
            print('class:', predicted[i], 'actual:', actual[i])
            
def mpg_to_rating(mpg):
    for i in range(len(mpg)):
        mpg[i] = rating(mpg[i])
    return mpg

def rating(mpg):
    if mpg < 14:
        return 1
    elif mpg < 15:
        return 2
    elif mpg < 17:
        return 3
    elif mpg < 20:
        return 4
    elif mpg < 24:
        return 5
    elif mpg < 27:
        return 6
    elif mpg < 31:
        return 7
    elif mpg < 37:
        return 8
    elif mpg < 45:
        return 9
    return 10

def folds_to_train(x, y, train_folds, test_folds):
    X_train = []
    y_train = []
    for row in train_folds:
        for i in row:
            X_train.append(x[i])
            y_train.append(y[i])

    X_test = []
    y_test = []
    for row in test_folds:
        for i in row:
            X_test.append(x[i])
            y_test.append(y[i])

    return X_train, y_train, X_test, y_test

def add_config_stats(matrix):
    del matrix[0]
    for i,row in enumerate(matrix):
        row[0] = i+1
        row.append(sum(row))
        row.append(round(row[i+1]/row[-1]*100,2))
        
def titanic_matrix(matrix):
    for i,row in enumerate(matrix):
        row.append(sum(row))
        row.append(round(row[i]/row[-1]*100,2))
        row.insert(0, i+1)
    matrix.append(['Total', matrix[0][1]+matrix[1][1], matrix[0][2]+matrix[1][2], matrix[0][3]+matrix[1][3], \
                   round(((matrix[0][1]+matrix[1][2])/(matrix[0][3]+matrix[1][3])*100),2)])

def print_tabulate(table, headers):
    print(tabulate(table, headers, tablefmt="rst"))

def all_same_class(instances):
    #grab first class label from that col
    prev_label = instances[0][-1]
     #loop through each row and compare class label with first
    for row in instances:
        if row[-1] != prev_label:
            #if diff label found, return false
            return False
    return True

#accepts a list of probabilities for each class label
def entropy(p_class_labels):
    E = 0
    for p in p_class_labels:
        if p == 0:
            continue
        E -= (p * math.log(p, 2))
    return E

def select_attribute(instances, available_attributes, header):
    ##entropy
    #get all possible class labels
    class_col = get_col_byindex(instances, -1)
    class_labels, counts = get_freq_str(class_col)
        #calculate E_start
    p_start_list = []
    for count in counts:
        p_start = count/len(instances)
        p_start_list.append(p_start)
    E_start = entropy(p_start_list)

    #create list to hold all E_new's to compare
    E_new_list = []
    #loop through each attr
    
    for attr in available_attributes:
        #get the attr col
        for header_attr in header:
            if attr == header_attr:
                attr_index = header.index(header_attr)
        col =  get_col_byindex(instances, attr_index)
        #find all possible domains
        domains, domain_counts = get_freq_str(col)
        #var to hold all calculated E's for the domains of this attr
        E_list = []
        #for each domain in that attr
        for i in range(len(domains)):
            domain = domains[i]
            #var to hold all P's
            p_list =[]
            #loop through each class label
            for label in class_labels:
                # create var to hold the count for number of rows that are C&V
                count = 0
                for j in range(len(instances)):
                    #if domain & label then count++
                    if str(instances[j][attr_index]) == str(domain) and str(instances[j][-1]) == str(label):
                        count += 1
                #calulate the P for that class label and domain = (Num of inst w this domain and this class label / Tot Num of inst of this domain)
                P = count/domain_counts[i]
                #add this to p_list for this domain
                p_list.append(P)
            #get E for this domain = entropy(p_list)
            E = entropy(p_list)
            #add this E to E_list for this attr
            E_list.append(E)
        #find E_new = (Sum: (count with this domain / tot of all instances)*(E_list))
        E_new = 0
        for k in range(len(domains)):
            E_new += (domain_counts[k]/len(instances)) * E_list[k]
        #append this to E_new_list
        E_new_list.append(E_new)
    #compare each E_new to E_start return the one furthest
    info_gain = 0
    for e_index in range(len(E_new_list)):
        if E_start - E_new_list[e_index] > info_gain:
            split_index = e_index
            info_gain = E_start - E_new_list[e_index]

    return available_attributes[split_index]

def partition_instances(instances, split_attribute, attr_domains, header):
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
    attribute_domain = attr_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0
    # lets build a dictionary
    partitions = {} # key (attribute value): value (list of instances with this attribute value)
    # task: try this!
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

def tdidt(current_instances, available_attributes, attr_domains, header):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes, header)
    # print("splitting on:", split_attribute)
    available_attributes.remove(split_attribute)
    new_atts = available_attributes
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attr_domains, header)
    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        # print("working with partition for:", attribute_value)
        value_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            # print("CASE 1")
            #get the class label (the y_train val) 
            label = partition[0][-1]
            #append value_subtree with ["Leaf", class label,  num_of_instances, total_num]
            value_subtree.append(["Leaf", label, len(partition), len(current_instances)])
            tree.append(value_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(new_atts) == 0:
            # print("CASE 2")
            stats = compute_partition_stats(partition, -1)
            stats.sort(key=lambda x: x[1])
            label = stats[-1][0]
            value_subtree.append(["Leaf", label, len(partition), len(current_instances)])
            tree.append(value_subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            # print("CASE 3")
            stats = compute_partition_stats(current_instances, -1)
            stats.sort(key=lambda x: x[1])
            label = stats[-1][0]
            return ["Leaf", label, len(partition), len(current_instances)]


        else: # all base cases are false... recurse!!
            subtree = tdidt(partition, new_atts, attr_domains, header)
            #append subtree to value_subtree
            value_subtree.append(subtree)
            #after handling case append subtree to tree
            tree.append(value_subtree)
    
    return tree

def build_header(X_train):
    row = X_train[0]
    header =[]
    for i in range(len(row)):
        attr = "attr" + str(i)
        header.append(attr)
    return header

def get_attr_domains(X_train, header):
    #loop through each column and get a list of every domain in tha col
    attr_domains = {}
    for i in range(len(header)):
        col = get_col_byindex(X_train, i)
        #vals is a list of every domain in that col
        vals, counts = get_freq_str(col)
        attr_domains[header[i]] = vals
    return attr_domains

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        # now I need to find which "edge" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match!! recurse!!
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label

def compute_partition_stats(instances, class_index):
    stats = {}
    for x in instances:
        if x[class_index] in stats:
            stats[x[class_index]] += 1
        else:
            stats[x[class_index]] = 1

        stats_array = []
        for key in stats:
                stats_array.append([key, stats[key], len(instances)])
        
    return stats_array
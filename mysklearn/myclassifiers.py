"""
Charles Walker 
CPSC 322
Section 02
PA6
"""
import mysklearn.myutils as myutils
import numpy as np
import copy
import operator
import random

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        x = []
        for sample in X_train:
            x.append(sample[0])

        y = y_train
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
        b = mean_y - m * mean_x 
        self.intercept = b
        self.slope = m 
        pass # TODO: copy your solution from PA4 here

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        x = []
        for sample in X_test:
            x.append(sample[0])
        
        y = []
        for i in range(len(x)):
            y_val = round((self.slope * x[i]) + self.intercept, 5)
            y.append(y_val)
        return y # TODO: copy your solution from PA4 here


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        #enumerate returns pairs, first element is index second element is elememnt
        #from knn example in CLassificationFun
        train = copy.deepcopy(self.X_train)
        k = self.n_neighbors

        all_distances = []
        all_neighbor_indices = []
        for test in X_test:
            for i, instance in enumerate(train):
                # append the class label
                instance.append(self.y_train[i])
                # append the original row index
                instance.append(i)
                # append the distance to [2, 3]
                dist = myutils.compute_euclidean_distance(instance[:2], test)
                instance.append(dist)
            
            # sort train by distance
            train_sorted = sorted(train, key=operator.itemgetter(-1))

            # grab the top k
            top_k = train_sorted[:k]
            dists = []
            indices = []
            for instance in top_k:
                dists.append(instance[-1])
                indices.append(instance[-2])
            all_distances.append(dists)
            all_neighbor_indices.append(indices)
        
        return all_distances, all_neighbor_indices # TODO: fix this


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        dists, all_indices = self.kneighbors(X_test)

        predicted_y_vals = []
        for indices in all_indices:
            y_vals = []
            for index in indices:
                y_vals.append(self.y_train[index])
            values, counts = myutils.get_freq_1col(y_vals)

            index_avg, avg = max(enumerate(counts), key=operator.itemgetter(1))
            
            predicted_y_vals.append(values[index_avg])
        
        return predicted_y_vals # TODO: fix this

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.

        
        Priors
            1. Probability of C (label)
                P(C) = Total/NumOfInsancesOfC
                ex: #C/Total
            2. Probability of X, (instance/row)
        Posteriors
            1. Probability of row given class label
                use independence assumption
                P(X|C) = P(V1 and C) * P(V2 and C) *...etc
                    P(V|C) = (#C&V/TotalLenTable)/P(C)
                    **only for categorical
            2. Probability of class label given row
                P(C|X) = P(X|C)*P(C)

        
        """

        #priors

        #get each class
        self.priors = []
        self.posteriors = []
        if isinstance(y_train[0], int):
            c_list, counts = myutils.get_freq_1col(y_train)
        else:
            c_list, counts = myutils.get_freq_str(y_train)

        #create list of priors objects, [label, probability], add to priors
        for i in range(len(c_list)):
            p = counts[i] / len(y_train)
            prior = [c_list[i], p]
            self.priors.append(prior)

        #posteriors
        
        #calculate probability of V and C for every possible V for each col (excluding c col)
        #loop through each col
        for i in range(len(X_train[0])):
            col = myutils.get_col_byindex(X_train, i)

            #############check if values in col are categorical or coninuous
            

                #get a list of every possible value and their counts (get_freq)
            val_list, counts = myutils.get_freq_str(col)
                #create list of posterior objects, [value, probability], add to this col's posteriors list

            #create list to hold all posteriors for col
            col_posteriors = [i]
            #loop through each C
            for c_index in range(len(c_list)):
                #create list to hold P(V|C)'s for this class
                posteriors = [c_list[c_index]]
                #loop through each V
                for V in val_list:
                    # create var to hold the count for number of rows that are C&V
                    count = 0     
                    #loop through each row
                    for j in range(len(X_train)):
                        #if C&V then count++
                        if str(X_train[j][i]) == str(V) and str(y_train[j]) == str(c_list[c_index]):
                            count += 1
    
                    # calc P(V|C) = count/Total#Rows
                    p = count/len(y_train)

                    p = p/self.priors[c_index][1]

                    # make [V_name, P] obj
                    posterior = [V, p]
                    #append obj to list of P(V|C)'s for this class
                    posteriors.append(posterior)
                col_posteriors.append(posteriors)
            #append col_posteriors, [col_index, [class_label, [val_name, P] ] ], to self.posteriors
            self.posteriors.append(col_posteriors)
        pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        
        Predict
            given X, find C such that P(C|X) is greatest
            1. Calculate all P(Ci|X)
            2. compare them
        """
        #calc P(X|C) by multiplying every corresponding posterior for each col val

        c_list = myutils.get_col_byindex(self.priors, 0)
        y_predicted = []
        #loop through x_test rows
        for row in X_test:
            # each x_test is a row with specific values. Compute the total prob for each possible class label
            #all_p_cx = [] : list to hold all P(C|X) to compare
            all_p_cx = []
            #loop through each class label
            for c_list_index in range(len(c_list)):
                #p_cx: this is P(C|X)
                p_cx = 0 
                #loop through each val in the row
                for curr_val_index in range(len(row)):
                    ##find the posterior for that val in that col
                    #loop through self.posteriors
                    post_found = False
                    for posteriors in self.posteriors:
                        if post_found == True:
                            break
                        #if curr_col_index == self.posteriors[curr_index:A][0]
                        if curr_val_index == posteriors[0]:
                            post_found = True   
                            #loop through self.posteriors[A]
                            for i in range(len(posteriors)):
                                if i == 0:
                                    continue
                                #if posterior.class == curr class label C
                                if str(posteriors[i][0]) == str(c_list[c_list_index]):
                                    #loop through the list with that C
                                    for j in range(len(posteriors[i])):
                                        #if posterior.val == given attr Val
                                        if j == 0:
                                            continue
                                        if str(posteriors[i][j][0]) == str(row[curr_val_index]):
                                            p = posteriors[i][j][1]
                                            if p_cx == 0:
                                                p_cx = 1
                                            p_cx = p_cx*p
                                            break
                p_cx = p_cx*self.priors[c_list_index][1]
                #append p_cx to all_p_cx
                all_p_cx.append(p_cx)
            #compare each p_cx from that list and find the index of max
            best_p_index = all_p_cx.index(max(all_p_cx))         
            #append the class label with corresponding index to y_predicted
            y_predicted.append(c_list[best_p_index])
        return y_predicted

class MyZeroClassifier:
    def __init__(self):
        """Initializer for MyZeroClassifier.
            only takes y_train
            Zero-R: classifies an instance using "zero rules"... 
            it always predicts the most common class label in the training set. 
            For example, if 99% of the dataset is positive instances, it always predicts positive.
        """
        self.y_train = None

    def fit(self, y_train):
        self.y_train = y_train
        pass
    
    def predict(self):
        vals, counts = myutils.get_freq_str(self.y_train)
        i = counts.index(max(counts)) 
        y_predict = vals[i] 
        return y_predict

class MyRandomClassifier:
    def __init__(self):
        """Initializer for MyRandomClassifier.
            Random classifier: classifies an instance by randomly choosing a class label 
            (class label probabilities of being chosen are weighted based on their frequency in the training set).
        """
        
        self.y_train = None

    def fit(self, y_train):
        self.y_train = y_train
        pass

    def predict(self): 
        vals, counts = myutils.get_freq_str(self.y_train)
        p_list = []
        for count in counts:
            curr_p = count/len(self.y_train)
            p_list.append(curr_p)

        pred = np.random.choice(vals, p= p_list)

        return pred

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        # compute a "header" ["att0", "att1", ...]
        header = myutils.build_header(X_train)
        # compute the attribute domains dictionary
        attr_domains = myutils.get_attr_domains(X_train, header)
        # my advice is to stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # initial call to tdidt current instances is the whole table (train)
        available_attributes = header.copy() # python is pass object reference

        self.tree = myutils.tdidt(train, available_attributes, attr_domains, header)
        # print("tree:", self.tree)
        pass # TODO: fix this
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # APIServiceFun interview_app.py
        #lecture 4/8
        y_predicted = []
        header = myutils.build_header(self.X_train)
    
        for instance in X_test:
            y_predicted.append(myutils.tdidt_predict(header, self.tree, instance))
        
        # print(y_predicted)

        return y_predicted # TODO: fix this

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """

        #traverse the tree
            #save first attr name from [Attribute, attr_name, ...]
        
        pass # TODO: fix this

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

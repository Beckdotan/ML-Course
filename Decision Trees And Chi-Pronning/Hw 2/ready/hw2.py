import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # get the labels from the last column
    labels = data[:, -1]  
     # count the amount of each label
    _, counts = np.unique(labels, return_counts=True)
    # calculate the probability of each label
    probs = counts / len(labels)  
    # calculate the Gini impurity using the actual equation. 
    gini = 1 - np.sum(probs ** 2)  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    #same as gini
    labels = data[:, -1] 
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels) 
    #different equation:
    entropy = -np.sum(probs * np.log2(probs)) 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    ############ Goodness of split ###########
    # Split the data into groups based on the feature value
    for row in data:
        value = row[feature]
        if value not in groups:
            groups[value] = []
        groups[value].append(row)


    total_instances = len(data)

    ## If wants to return the gain ratio instead of the goodness of split. 
    if gain_ratio:
        split_information = 0.0
        #calc split information
        for group in groups.values():
            proportion = len(group) / total_instances
            split_information -= proportion * np.log2(proportion)
        #calc information gain with entropy always. (recursive call)
        info_gain = goodness_of_split(data, feature, calc_entropy)
        #calc gain ratio
        goodness = info_gain[0] / split_information
        ##return since there is not need to continue in this case. 
        return goodness,groups
            

    #calculating the original impurity. 
    initial_impurity = impurity_func(data)

    for group in groups.values():
        proportion = len(group) / total_instances
        impurity = impurity_func(np.array(group))
        goodness += proportion * impurity

    goodness = initial_impurity - goodness



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels = self.data[:, -1]
        label_counts = {}
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        pred = max(label_counts, key=label_counts.get)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
         # Check if the current node is a terminal node
        if len(np.unique(self.data[:, -1])) == 1 or self.depth >= self.max_depth:
            self.terminal = True
            return
        
        # Find the best feature to split according to
        best_feature = None
        best_gain = -np.inf 
        for i in range(self.data.shape[1] - 1):
            gain = goodness_of_split(self.data, i, impurity_func, self.gain_ratio)
            if gain[0] > best_gain:
                best_feature = i
                best_gain = gain[0]
        
        
        ##if chi = 1 than dont run it
        # Find if chi test is ok 
        if  self.chi != 1 and chi_squared_test(self.data, self.chi, best_feature):
            self.terminal = True
            return
        
        self.feature = best_feature

        # Create a child node for each unique value of the best feature
        for val in np.unique(self.data[:, best_feature]):
            child_data = self.data[self.data[:, best_feature] == val, :]
            child_node = DecisionNode(child_data, depth=self.depth + 1,
                                    chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child_node, val)

            # Recursively call the split function for each child node
            child_node.split(impurity_func)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root = DecisionNode(data,-1,0,chi,max_depth,gain_ratio)
    root.split(impurity)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    current_node = root
    while not current_node.terminal:
        #determine the feature index and find the corresponding value of the instance
        feature_index = current_node.feature
        node_value = instance[feature_index]
        
        #find the index of the node we want to proceed to, and update it
        for i in range(len(current_node.children_values)):
            if node_value == current_node.children_values[i]:
                current_node = current_node.children[i]
                break
            
            #handle the case where the feature value of the instance is not equal to any 
            #of the values that were used to create the children of the current node
            if i==len(current_node.children_values) -1:
                current_node = current_node.children[-1]

        #Update the feature index for the next internal node
        feature_index = current_node.feature

    return current_node.calc_node_pred()


    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    #count the correctly predicted instances
    counter = 0
    for row in dataset:
        predicted_label = predict(node, row)
        if predicted_label == row[-1]:
            counter += 1

    accuracy = (counter / len(dataset)) * 100
    return accuracy
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    for max in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(data = X_train, impurity = calc_entropy, gain_ratio=True ,max_depth = max)

        train_accuracy = calc_accuracy(tree, X_train)
        training.append(train_accuracy)

        test_accuracy = calc_accuracy(tree, X_test)
        testing.append(test_accuracy)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for chi in [1, 0.5,0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(X_train, calc_entropy, gain_ratio=True,  chi=chi)

        train_accuracy = calc_accuracy(tree, X_train)
        chi_training_acc.append(train_accuracy)

        test_accuracy = calc_accuracy(tree, X_test)
        chi_testing_acc.append(test_accuracy)

        depth.append(tree.depth)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_nodes = 1
    for child in node.children:
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes



def chi_squared_test(data, chi_parameter, best_feature):
        """
        This function performs the chi-squared test to determine whether a split should happen or not.

        Input:
        - data: the dataset to split
        - chi_parameter: the critical value for the chi-squared test
        - best_feature: the index of the best feature to split on

        Returns:
        - True if the split should not happen based on the chi-squared test, False otherwise
        """
    
        #for val in np.unique(data[:, best_feature]):
         #   child_data = data[data[:, best_feature] == val, :]
        #DF = degree of freedom -> number of uniq values in the feature -1. * number of poosible value in label -1. 
        #alpha - is the chi.parameter.-> self.chi
        # check with the table if the chiparameter higher then the one we got, if yes dont do child. 
        #chi_table[df][self.chi] > calculate than dont split. (return true). 

        # calc degree of freedom
        degree_of_freedom = (len(np.unique(data[:, best_feature])) -1)  * (len(np.unique(data[:, -1])) -1)

        # calc  Params for equation
        #values = data[best_feature].value_counts()
        values =np.unique(data[:, best_feature])
        probability_y_e = sum(1 for row in data if row[-1] == 'e') / sum(1 for row in data)
        probability_y_p = 1-probability_y_e

        chi_squared = 0

        for value in values:#.iteritems():
            df = sum(1 for row in data if row[best_feature] == value)
            pf = sum(1 for row in data if row[best_feature] == value and row[-1] == 'p')
            nf = sum(1 for row in data if row[best_feature] == value and row[-1] == 'e')

            e_poisonous = df * probability_y_p
            e_edible = df * probability_y_e

            chi_squared += ((pf - e_poisonous)**2 / e_poisonous) + ((nf - e_edible)**2 / e_edible)

        # if (chi_table[df][self.chi] > eq  return true else false; )
        if chi_table[degree_of_freedom][chi_parameter] > chi_squared: 
            return True
        return False
               


       
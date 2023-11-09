###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    mean_X = np.mean(X, axis=0)
    range_X= np.max(X, axis=0) - np.min(X, axis=0)
    X = (X-mean_X)/ range_X
    
    mean_y = np.mean(y, axis=0)
    range_y= np.max(y, axis=0) - np.min(y, axis=0)
    y = (y-mean_y)/ range_y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    # Add a column of ones to X using np.append()
    ones_arr = np.ones((X.shape[0]))
    X = np.c_[ones_arr, X]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    m = len(y)  # number of instances
    J = np.sum((X.dot(theta) - y)**2) / (2*m)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = len(y)  # number of instances
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    ## ASSUMING:
    ## 1. X isn't singular (since we will expect atlist one feature)
    ## 2. M > N -> there will be more raws than features. 
    for i in range(num_iters):
        y_pred = X.dot(theta)
        theta = theta - (alpha / m) * X.T.dot( y_pred - y)
        J_history.append(compute_cost(X, y, theta))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    
    # Compute the pseudoinverse of X using X+ = (X.T @ X)^(-1) @ X.
    X_pinv = np.linalg.inv(X.T @ X) @ X.T
    
    # Compute the optimal parameters using pinv_theta = X+ @ y.
    pinv_theta = X_pinv @ y
    	
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    ## ASSUMING:
    ## 1. X isn't singular (since we will expect atlist one feature)
    ## 2. M > N -> there will be more raws than features. 
    m = len(y)  # number of instances
    prev_loss = float('inf')
    for i in range(num_iters):
        h = np.dot(X, theta)
        error = h - y
        gradient = np.dot(X.T, error) / len(y)
        theta = theta - alpha * gradient
        loss = compute_cost(X, y, theta)
        J_history.append(loss)
        if prev_loss - loss < 1e-8:
            break
        prev_loss = loss
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    for alpha in alphas:
        theta = np.ones(X_train.shape[1])
        theta, hist = efficient_gradient_descent(X_train, y_train, theta, alpha, iterations)
        J_val = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = J_val
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    m, n = X_train.shape
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################    
    for i in range(5):
        print('selected_features are: ' , selected_features)
        current_selected_feature = 0
        current_minimal_cost = float('inf')
        
        for j in range(n): ## j is feature
            
            #prepossesing
            candidate_features = selected_features + [j]
            X_train_subset = X_train[:, candidate_features]
            X_val_subset = X_val[:, candidate_features]
            #print('candidate_features are: ' , candidate_features)
            #print(X_val_subset)
            #print(X_val_subset)
            
            # Init parameters
            theta = np.zeros(len(candidate_features))
            #print(theta)
            
            # Run efficient gradient descent to obtain optimal parameters
            theta, cost_history = efficient_gradient_descent(X_train_subset, y_train, theta, best_alpha, iterations)
            #print('after learning')
            
            # Compute accuracy on validation set
            cost = compute_cost(X_val_subset, y_val, theta)
            
            if cost < current_minimal_cost: 
                current_minimal_cost = cost
                current_selected_feature = j

        selected_features.append(current_selected_feature)
        
            
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    feature_cols = df_poly.columns
    num_features = len(feature_cols)
    for i in range(num_features):
        for j in range(i, num_features):
            feature_1 = feature_cols[i]
            feature_2 = feature_cols[j]
            if( i == j):
                interaction_feature = feature_1 + "^2"
            else: 
                interaction_feature = feature_1 + "*" + feature_2
            df_poly[interaction_feature] = df[feature_1] * df[feature_2]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
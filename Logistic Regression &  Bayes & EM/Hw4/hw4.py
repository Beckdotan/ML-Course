import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        #Adding ones column as first column to X for bias. 
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        
        # Inint Theta:
        self.theta = np.random.rand(X.shape[1])


    
        for iteration in range(self.n_iter):
            #append current thetas
            self.thetas.append(self.theta)

            #helper value
            product = np.dot(X, self.theta)
            
            #sigmoid function
            sig = 1.0 / (1 + np.exp(-product))
            
            #J(theta) helper value
            errors = -y * np.log(sig) - (1-y) * np.log(1-sig)
            
            #J(theta) = cost = mean of errors vector
            cost = np.sum(errors)

            #if cost difference is not relevant
            if len(self.Js) > 0 and abs(self.Js[-1] - cost) < self.eps:     
                self.Js.append(cost) 
                break 
            
            self.Js.append(cost)


            ## update theta:
            gradient = (-self.eta * np.dot(X.T,(sig - y))) / y.size
            self.theta = self.theta + gradient
             
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
         #Adding ones column as first column to X for bias. 
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        
        product = np.dot(X, self.theta)
        sig = 1.0 / (1+ np.exp(-product))

        preds = np.where(sig > 0.5, 1 , 0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    accuracies = []

    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    fold_size = len(X) // folds
    X_folds = np.array_split(X_shuffled, folds)
    y_folds = np.array_split(y_shuffled, folds)

    for i in range(folds):
        # Create training and validation sets
        X_train = np.concatenate(X_folds[:i] + X_folds[i + 1 :])
        y_train = np.concatenate(y_folds[:i] + y_folds[i + 1 :])
        X_val = X_folds[i]
        y_val = y_folds[i]

        algo.fit(X_train, y_train)
        preds = algo.predict(X_val)

        #accuracy
        accuracy = np.sum(preds==y_val) / y_val.size
        accuracies.append(accuracy)

    cv_accuracy = np.mean(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    exponent = -0.5*(((data-mu)/sigma) ** 2)
    p = np.exp(exponent) / (np.sqrt(2*np.pi) * sigma)

    
    
   # a = np.exp(-0.5 * np.power((data - mu) / sigma, 2))
   # b = sigma * np.sqrt(2 * np.pi)
    
    #return (a / b).T
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        # Initialize weights equally
        self.weights = np.ones(self.k) / self.k

        # the means are initialized randomly by randomly selecting k data points from the input data
        number_samples = data.shape[0]
        random_indices = np.random.choice(number_samples, self.k)
        self.mus = data[random_indices].flatten()

        # Initialize standard deviations to 1 for all
        #number_features = data.shape[1]
        self.sigmas = np.ones(self.k)

        # Initialize responsibilities to be 0 matrix.  
        # note: The value at coordinate (i, j) in the responsibilities matrix represents 
        #the posterior probability of the i-th data point belonging to the 
        #j-th Gaussian component
        self.responsibilities = np.zeros((number_samples, self.k))

        # Initialize costs
        self.costs = []

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities

        
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        number_samples = data.shape[0]

        for i in range(number_samples):
            #get the i'th  sample inn order to update its value in the responsibilities matrix
            x = data[i]
            for j in range(self.k):
                # Calculate  p( x | j'th distribution)
                p = norm_pdf(x, self.mus[j], self.sigmas[j])
                # Calculate the posterior, that is p(j'th distribution | x ))
                self.responsibilities[i, j] = self.weights[j] * p

            # Normalize the responsibilities for the i-th data point
            self.responsibilities[i] /= np.sum(self.responsibilities[i])


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        GETS ONE COLUMN (FEATURE AT A TIME)
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # Update the weights
        self.weights = np.mean(self.responsibilities, axis=0)

        for j in range(self.k):
            
            # Update the means
            self.mus[j] = np.sum(self.responsibilities[:, j].reshape(-1, 1) * data, axis=0) / (self.weights[j] * data.shape[0])

            # Update the standard deviations
            self.sigmas[j] = np.sqrt(np.sum(self.responsibilities[:, j].reshape(-1, 1) * (data - self.mus[j])**2, axis=0) /  (self.weights[j] * data.shape[0]))

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.init_params(data)

        self.costs.append(self.calculate_cost(data)) 

        for _ in range(self.n_iter):
          self.expectation(data)
          self.maximization(data)

          self.costs.append(self.calculate_cost(data))

          if abs(self.costs[-2] - self.costs[-1]) < self.eps:
            break

        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    def calculate_cost(self, data):
      """
      Calculate the log-likelihood cost for the current params and data.
      """
      
      likelihoods = np.zeros((data.shape[0] , 1))
      for j in range(self.k):
        prob_x = norm_pdf(data, self.mus[j], self.sigmas[j])
        likelihoods += self.weights[j] * prob_x

      cost = np.sum(np.log(likelihoods))
      return cost

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = np.zeros(data.shape[0])
    for i in range(mus.shape[0]):
      pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.means = None
        self.variances = None
        self.weights = None


    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n_samples = X.shape[0]
        n_features = X.shape[1]

        classes = np.unique(y)
        n_classes = len(classes)

        self.means = np.zeros((n_classes, n_features , self.k))
        self.variances = np.zeros((n_classes, n_features, self.k))
        self.weights = np.zeros((n_classes, n_features, self.k))

        # Prior probabilities vector for all classes 
        self.prior = np.zeros(n_classes)  
        for current_class in classes:
            self.prior[current_class] = np.count_nonzero(y == current_class) / n_samples

        # compute gaussian params per feature and class
        em = EM(self.k)

        for current_class in classes:
            for feature_index in range(n_features):
                em.fit((X[y == current_class][:,feature_index]).reshape(-1,1))
                current_weights, current_means, current_variances = em.get_dist_params()
                self.means[current_class][feature_index] = current_means
                self.variances[current_class][feature_index] = current_variances
                self.weights[current_class] = current_weights
                
        # for feature_index in range(n_features):
        #     for current_class in classes:
        #         em.fit((X[y == current_class][:,feature_index]).reshape(-1,1))
        #         current_weights, current_means, current_variances = em.get_dist_params()
        #         self.means[current_class][feature_index] = current_means.reshape(1,-1)[0]
        #         self.variances[current_class][feature_index] = current_variances.reshape(1,-1)[0]        
        #         self.weights[current_class] = current_weights.reshape(1,-1)[0]     

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = np.zeros(X.shape[0])
        results = np.zeros((len(self.means[0]), X.shape[0]))

        n_classes = self.means.shape[0]

        classes_results = np.zeros((n_classes , X.shape[0]))
        for curr_class in range(n_classes):
            sum_features = 1
            for feature in range(X.shape[1]):
                 p = gmm_pdf(X[:,feature], self.weights[curr_class][feature], self.means[curr_class][feature], self.variances[curr_class][feature])
                 sum_features *= p
            classes_results[curr_class] = sum_features * self.prior[curr_class]
        preds = np.argmax(classes_results, axis=0)

        # for clss in range(len(self.means[0])):
        #     #result[group] *= self.prior[group]
        #     group_result = np.zeros(X.shape[0])
        #     for group in range(self.k):
        #       result = np.zeros(X.shape[0])
        #       for feature in range(X.shape[1]):
        #            result += norm_pdf(X[ : , feature], self.means[feature][clss][group], self.variances[feature][clss][group])
        #       #n_features, len(unique_values), self.k
        #       group_result += self.weights[clss][group] * result
        #     group_result *= self.prior[clss]
        #     results[clss] = np.log(group_result)
        # #return preds as array that predicts the label that habvve the maximum value in results for each row
        # preds = np.argmax(results, axis=0)

        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    # Fit Logistic Regression model
    logistic_regression = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression.fit(x_train, y_train)
    lor_train_acc = calcAcc(logistic_regression.predict(x_train), y_train)
    lor_test_acc = calcAcc(logistic_regression.predict(x_test), y_test)

    # Fit Naive Bayes model
    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    predictions = naive_bayes.predict(x_train)
    bayes_train_acc = calcAcc(naive_bayes.predict(x_train), y_train)
    bayes_test_acc = calcAcc(naive_bayes.predict(x_test), y_test)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    np.random.seed(42)

    # Number of data points per class
    n_samples = 1000

    # Class 1: Gaussian distributions
    mean1_1 = [0, 0, 0]
    mean1_2 = [20, 20, 20]
    cov = np.eye(3)  # Covariance matrix (identity matrix for independence)

    class1_1 = np.random.multivariate_normal(mean1_1, cov, size=n_samples)
    class1_2 = np.random.multivariate_normal(mean1_2, cov, size=n_samples)

    # Class 2: Gaussian distributions
    mean2_1 = [10, 10, 10]
    mean2_2 = [30, 30, 30]

    class2_1 = np.random.multivariate_normal(mean2_1, cov, size=n_samples)
    class2_2 = np.random.multivariate_normal(mean2_2, cov, size=n_samples)

    # Combine the two classes
    dataset_a_features = np.vstack((class1_1, class1_2, class2_1, class2_2))
    dataset_a_labels = np.hstack((np.zeros(2 * n_samples, dtype=int), np.ones(2 * n_samples, dtype=int)))

    # Class 1: Gaussian distributions
    mean1_1 = [0, 0, 0]
    mean1_2 = [0, 20, 50]
    cov = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])  # Covariance matrix with correlations

    class1_1 = np.random.multivariate_normal(mean1_1, cov, size=n_samples)
    class1_2 = np.random.multivariate_normal(mean1_2, cov, size=n_samples)

    #Class 2: Gaussian distributions
    mean2_1 = [0, 20, 0]
    mean2_2 = [20, 20, 50]


    class2_1 = np.random.multivariate_normal(mean2_1, cov, size=n_samples)
    class2_2 = np.random.multivariate_normal(mean2_2, cov, size=n_samples)

    # Combine the two classes
    dataset_b_features = np.vstack((class1_1, class1_2, class2_1, class2_2))
    dataset_b_labels = np.hstack((np.zeros( 2 * n_samples, dtype=int), np.ones(2 * n_samples, dtype=int)))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ##########################################################################


    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }

def calcAcc (pred, y):
    arr = np.ones(len(pred))
    arr = arr[pred == y]
    return len(arr) / len(pred)

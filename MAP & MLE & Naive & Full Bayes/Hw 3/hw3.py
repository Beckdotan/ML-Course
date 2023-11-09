import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)
        
        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.25,
            (0, 1): 0.05,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.25,
            (0, 1): 0.05,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(Y=y, C=c)
#p(x&y)*p(c)
        
        # p(x.y.c) = p(x,c) * p(y)
        
        self.X_Y_C = {
            (0, 0, 0): 0.125,
            (0, 0, 1): 0.005,
            (0, 1, 0): 0.125,
            (0, 1, 1): 0.045,
            (1, 0, 0): 0.125,
            (1, 0, 1): 0.045,
            (1, 1, 0): 0.125,
            (1, 1, 1): 0.405
        }  # P(X=x, Y=y, C=c)
        
    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for (x,y) in X_Y:
            if not np.isclose(X_Y[(x,y)], X[x]*Y[y]):
                return True
        return False

        

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """

        p(x,y|c) = p(x|c)*p(y|c) <=> p(x,y,c) = p(x,c)*p(y,c)
        P(x,y|c) = p(x,y,c)/p(c)
        p(x|c) = p(x,c) / p(c)
        p(y|c) = p(y,c) / p(c)
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for (x,y,c) in X_Y_C:
             if not np.isclose(X_Y_C[(x,y,c)], X_C[(x,c)] * Y_C[(y,c)]):
                #print((x,y,c))
                return False
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p = rate**k * np.exp(-rate) / np.math.factorial(k)
    log_p = np.log2(p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    likelihoods = np.zeros(len(rates))
    for idx, rate in enumerate(rates):
        rate_sum = 0
        for sample in samples:
            rate_sum += poisson_log_pmf(sample, rate)
        likelihoods[idx] = rate_sum
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    max_index = np.argmax(likelihoods)
    rate = rates[max_index]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    sum = 0
    mean = np.sum(samples) / len(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################

    #p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    p = (1 / ( np.sqrt(2 * np.pi * (std**2)))) * np.exp(-(((x - mean) ** 2) / (2 * (std ** 2))))
    ## p = np.exp((-(x-mean)**2)/(2*std**2))/np.sqrt(2*np.pi*std**2)
    #print(p)
    ###########################################################################
    #                            END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.dataset = dataset
        self.class_value = class_value

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        #class_ds = [x for x in self.dataset[:,-1] if self.dataset[:,-1] == self.class_value]
        class_ds = self.dataset[self.dataset[:,-1] == self.class_value]
        class_ds_length = class_ds.shape[0]
        full_data_length = self.dataset.shape[0]
        
        prior = class_ds_length / full_data_length
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        class_data = self.dataset[self.dataset[:, -1] == self.class_value]
        #mean = np.mean(class_data[:,:-1])
        #std = np.std(class_data[:,:-1])
        #normal_array = normal_pdf(x, mean, std)
        likelihood = 1.0
        
        for i in range(len(x) - 1):
            mean = np.mean(class_data[:, i])
            std = np.std(class_data[: ,i])
            likelihood *= normal_pdf(x[i], mean, std)
        
        #for value in normal_array:
         #  likelihood *= value
       ## likelihood = normal_array[0] * normal_array[1]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        #print("likelihhod")
        #print( self.get_instance_likelihood(x))
        #print(self.get_prior())
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior_0 = self.ccd0.get_instance_posterior(x)
        posterior_1 = self.ccd1.get_instance_posterior(x)
        if posterior_0 > posterior_1:  
            pred = 0
        else: 
            pred = 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    correctly_classified = 0
    
    for x in test_set:
        if map_classifier.predict(x) == x[-1]:
            correctly_classified+=1

    acc = correctly_classified / len(test_set[:,-1])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    #print(x.shape) # to check if the shape[0] is really the num of features. 
    d = x.shape[0] # number of dimensions
    det_cov = np.linalg.det(cov)  # determinant of the covariance matrix
    inv_cov = np.linalg.inv(cov)  # inverse of the covariance matrix
    diff = x[:-1] - mean  # difference between x and mean
    power = -0.5 * np.transpose(diff)  @ inv_cov @ diff
    # Compute the probability density function
    pdf = ((2*np.pi) ** (-d/2)) * (det_cov ** (-0.5)) * (np.exp(power))
    #print(pdf)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        class_ds = self.dataset[self.dataset[:,-1] == self.class_value]
        class_ds_length = class_ds.shape[0]
        full_data_length = self.dataset.shape[0]
        
        prior = class_ds_length / full_data_length
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        '''
        class_data = self.dataset[self.dataset[:, -1] == self.class_value]
        mean = np.mean(class_data)
        std = np.std(class_data)
        normal_array = normal_pdf(x, mean, std)
        likelihood = normal_array[0] * normal_array[1]
        '''
        class_data = self.dataset[self.dataset[:, -1] == self.class_value]
        #print(class_data)
        mean = np.mean(class_data[:,:-1], axis = 0)    
        cov = np.cov(np.transpose(class_data[:,:-1]))
        likelihood = multi_normal_pdf(x,mean,cov)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the prior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            pred = 0
        else: pred = 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x):
            pred = 0
        else: pred = 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

'''
1M-5 is not sick
5 sick

5 of the sick have green hair
non of the not sick have green hair. 

prior (0) = 1M-5/1M ~1
priror (1) ~ 0

liklihood to be with green hair given not sick -> 1/1M-5 ~ 0
Liklihhod to be with greaa hair given you are sick -> 5/5 = 1

MAP: 
    MAP (0) = 5/1M * 1 ~ 0
    MAP (1) = ~1 * ~0 ~~ 0


'''



class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        class_ds = self.dataset[self.dataset[:,-1] == self.class_value]
        class_ds_length = class_ds.shape[0]
        full_data_length = self.dataset.shape[0]
        
        prior = class_ds_length / full_data_length
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        class_data = self.dataset[self.dataset[:, -1] == self.class_value]
        likelihood = 1
        ni = class_data.shape[0]
        for feature in range(0,class_data[:,:-1].shape[1]):
            if self.dataset[self.dataset[:,feature] == x[feature]].shape[0] == 0:
                likelihood *= EPSILLON
            else:
                class_and_attibute_data = class_data[class_data[:,feature] == x[feature]]
                nij = class_and_attibute_data.shape[0]
                vj = len(np.unique(self.dataset[:,feature]))
                ij_likelihood = (nij+1) / (ni + vj)
                likelihood *= ij_likelihood

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior_0 = self.ccd0.get_instance_posterior(x)
        posterior_1 = self.ccd1.get_instance_posterior(x)
        if posterior_0 > posterior_1:  
            pred = 0
        else: 
            pred = 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        correctly_classified = 0
        
        for x in test_set:
            if self.predict(x[:-1]) == x[-1]:
                correctly_classified+=1

        acc = correctly_classified / len(test_set[:,-1])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc



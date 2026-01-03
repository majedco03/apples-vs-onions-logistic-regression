import numpy as np
from PIL import Image
import os

class Data():
    def __init__(self, value):
        self.value = value

# Load data method to load training and test datasets into numpy arrays
    @staticmethod
    def load_data():
        # Load training data (apples)
        train_apple_dir = 'train data/apple'
        train_images = []
        train_labels = []
        
        # Load apples
        for filename in os.listdir(train_apple_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(train_apple_dir, filename)
                img = Image.open(img_path).convert('RGB').resize((64, 64))  # Convert to RGB and resize to 64x64
                img_array = np.array(img)
                img_flat = img_array.flatten()
                train_images.append(img_flat)
                train_labels.append(1)  # label 1 for apples
        
        # Load onions
        train_onion_dir = 'train data/onion'
        for filename in os.listdir(train_onion_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(train_onion_dir, filename)
                img = Image.open(img_path).convert('RGB').resize((64, 64))  # Convert to RGB and resize to 64x64
                img_array = np.array(img)
                img_flat = img_array.flatten()
                train_images.append(img_flat)
                train_labels.append(0)  # label 0 for onions
        
        X_train = np.array(train_images).T  # shape (features, num_samples)
        y_train = np.array(train_labels).reshape(1, -1)  # shape (1, num_samples)
        
        # Load test data
        test_images = []
        test_labels = []
        
        # Load test apples
        test_apple_dir = 'test data/apple'
        for filename in os.listdir(test_apple_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(test_apple_dir, filename)
                img = Image.open(img_path).convert('RGB').resize((64, 64))  # Convert to RGB and resize to 64x64
                img_array = np.array(img)
                img_flat = img_array.flatten()
                test_images.append(img_flat)
                test_labels.append(1)  # label 1 for apples
        
        # Load test onions
        test_onion_dir = 'test data/onion'
        for filename in os.listdir(test_onion_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(test_onion_dir, filename)
                img = Image.open(img_path).convert('RGB').resize((64, 64))  # Convert to RGB and resize to 64x64
                img_array = np.array(img)
                img_flat = img_array.flatten()
                test_images.append(img_flat)
                test_labels.append(0)  # label 0 for onions
        
        X_test = np.array(test_images).T  # shape (features, num_samples)
        y_test = np.array(test_labels).reshape(1, -1)  # shape (1, num_samples)
        
        # Normalize pixel values to [0,1]
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        return X_train, y_train, X_test, y_test
    

    # Sigmoid function 
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # initialize weights and bias with zeros
    @staticmethod
    def initialize(dim):
        w = np.zeros((dim, 1))
        b = 0
        return w, b
    
    # Propagation function to compute cost and its gradients
    @staticmethod
    def propagation(w, b, X, Y):
        m = X.shape[1]

        #forward propagation (Computing cost)
        Activation = Data.sigmoid(np.dot(w.T,X)+b) # compute activation
        epsilon = 1e-8  # for numerical stability
        cost = - ((1/m) * np.sum( Y * np.log(Activation + epsilon) + (1-Y) * np.log(1-Activation + epsilon) ))

        #backward propagation (Computing gradients)
        dw = 1/m * np.dot(X, (Activation - Y).T)
        db = 1/m * np.sum(Activation - Y)

        cost = np.squeeze(cost)  #to make the cost in the same dimension

        # gradients dictionary
        gradients = {"dw": dw,
                 "db": db}
        
        return gradients, cost
    

    # Optimization function to update weights and bias
    @staticmethod
    def optimization(w, b, X, Y, num_iterations, learning_rate = 0.05, print_cost = False):
        costs = []
        
        for i in range(num_iterations):
            # calculate gradients and cost using propagate function
            gradients, cost = Data.propagation(w, b, X, Y)
            
            dw = gradients["dw"]
            db = gradients["db"]
            
            # update rule is theta = theta - learning_rate * gradient
            w = w - learning_rate * dw
            b = b - learning_rate * db
            
            # recording the costs
            if i % 100 == 0:
                costs.append(cost)
        
        params = {"w": w,
                  "b": b}
        
        gradients = {"dw": dw,
                 "db": db}
        
        return params, gradients, costs
    
    # train function to train the logistic regression model
    @staticmethod
    def train(X_train, Y_train, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        dim = X_train.shape[0]
        w, b = Data.initialize(dim)
        
        parameters, gradients, costs = Data.optimization(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        
        return parameters, costs
        
       


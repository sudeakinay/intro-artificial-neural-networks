#############################  MY NEURAL NETWORK MODEL (TRAIN) #############################
import numpy as np                      # allows scienftific computing                                   
import joblib                           # allows load models

class NeuralNetwork:
    def __init__(self, learning_rate, iterations, input_dimension, hidden_layers):
        self.learning_rate = learning_rate
        self.iterations = iterations
        if np.any(learning_rate <= 0):
            raise ValueError("learning_rate must be greater than zero")
        if  np.any(iterations <= 0):
            raise ValueError("number of iteration must be greater than zero")

        self.input_dimension = input_dimension
        self.hidden_layers = hidden_layers
        self.weights = np.array([np.random.randn(input_dimension)])
        self.bias = np.random.randn()

        self.weights_input_to_hidden = np.random.uniform(-1, 1, (input_dimension, hidden_layers))
        self.weights_hidden_to_output = np.random.uniform(-1, 1, (hidden_layers))

        pre_hidden = np.zeros(hidden_layers)
        post_hidden = np.zeros(hidden_layers)
        self.pre_hidden = pre_hidden
        self.post_hidden = post_hidden
        LR = 1
        self.LR = LR
       
    def _LReLU(self, x):
        x = np.array(x)
        if (x < 0).any():
            x = 0.01 * x 
        elif (x >= 0).any():
            x = x  
        return x
 
    def _LReLU_deriv(self, x):
        x = np.array(x)
        if (x < 0).any():
            x = 0.01 
        elif (x >= 0).any():
            x = 1 
        return x

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._LReLU(layer_1)
        prediction = layer_2
        return prediction

    def _computer_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._LReLU(layer_1)
        prediction = layer_2

        derror_dprediction =  (prediction - target)
        dprediction_dlayer1 = self._LReLU_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = ( derror_dprediction * dprediction_dlayer1 * dlayer1_dbias )
        derror_dweights = ( derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)
        
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - ( derror_dweights * self.learning_rate)
        return self.weights, self.bias

    def train(self, input_vectors, targets, iterations):

        for current_iteration in range(iterations) :
            for sample in range(len(input_vectors[:, 0])):
                for node in range(self.hidden_layers):
                    self.pre_hidden[node] = np.dot(input_vectors[sample, :], self.weights_input_to_hidden, node)
                    post_hidden = self._LReLU(self.pre_hidden[node])

                pre_hidden_O = np.dot(post_hidden, self.weights_hidden_to_output)
                post_hidden_O = self._LReLU(pre_hidden_O)     
                fatal_error = post_hidden_O - targets[sample]

            for hidden_node in range(self.hidden_layers):
                S_error = fatal_error * self._LReLU_deriv(pre_hidden_O)
                gradient_hidden_to_output = S_error * post_hidden[hidden_node]

                for input_node in range(input_dimension):
                    input_value = input_vectors[sample, input_node]
                    gradient_input_to_hidden = S_error * self.weights_hidden_to_output[hidden_node] * self._LReLU_deriv(self.pre_hidden[hidden_node] * input_value)

                    self.weights_input_to_hidden[input_node, hidden_node] -= self.LR * gradient_input_to_hidden
                self.weights_hidden_to_output[hidden_node] -= self.LR * gradient_hidden_to_output

            if current_iteration % 100 == 0: 
                total_error = 0
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
                    prediction = self.predict(data_point)       
                    error = np.square(prediction - target)      
                    total_error = total_error + error   
                
            for i in total_error:
                if i <= 10^(-5):
                    break  

        return total_error, self.weights, self.bias

    def test_function(self, test_data, final_weights, final_bias):
        test_data_pred = []
        prediction = np.dot(test_data, final_weights) + final_bias
        for i in range(len(test_data)):
            if prediction[i] <= 0.69 :     # optimizing results
                test_data_pred.append(0)
                print("Setosa")
            elif (prediction[i] > 0.69).any() and (prediction[i] <= 1.5).any() :
                test_data_pred.append(1)
                print("Versicolor")
            elif prediction[i] >= 1.5:
                test_data_pred.append(2)  
                print("Virginica")  
        return test_data_pred

    def new_data_prediction(self, new_data, final_weights, final_bias):

        new_data_result = np.dot(new_data, final_weights) + final_bias
        if new_data_result <= 0.69:         # optimizing results
            new_data_result = 0
            print("Setosa")               
        elif (new_data_result> 0.69) and (new_data_result <= 1.5):           
            new_data_result = 1 
            print("Versicolor")            
        elif new_data_result >= 1.5:
            new_data_result = 2
            print("Virginica") 
        return new_data_result

learning_rate = 0.01
iterations = 100000
input_dimension = 1
hidden_layers = 5


model = NeuralNetwork(learning_rate, iterations, input_dimension, hidden_layers)
joblib.dump(model, "my_first_model")
loaded_model = joblib.load("my_first_model")

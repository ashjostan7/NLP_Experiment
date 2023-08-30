import numpy as np
from rnn_utils import Softmax, Tanh, CrossEntropyLoss

class RNNModel:

    def __init__(self, input_dim, output_dim, hidden_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        parameters = self.__initialise_parameters(input_dim, output_dim, hidden_dim)

        self.Way, self.Wax, self.Waa, self.by, self.b = parameters
        self.softmax = Softmax()
        self.oparams = None 

    def __initialise_parameters(self,input_dim, output_dim, hidden_dim):

        den = np.sqrt(hidden_dim) # Not sure why to we use den.

        weights_y = np.random.randn(output_dim, hidden_dim) / den
        bias_y = np.zeros((output_dim,1))

        weights_ax = np.random.randn(hidden_dim, input_dim) / den
        weights_aa = np.random.randn(hidden_dim, hidden_dim) / den
        bias = np.zeros((hidden_dim,1))

        return weights_y, weights_ax, weights_aa, bias_y, bias

    def forward(self, input_X):

        self.input_X = input_X

        self.layers_tanh = [Tanh() for x in input_X]
        hidden = np.zeros((self.hidden_dim , 1))

        self.hidden_list = [hidden]
        self.y_preds = []

        for input_x, layer_tanh in zip(input_X, self.layers_tanh):
            # print(f'Wax: {type(self.Wax)}')
            # print(f'Waa: {self.Waa.shape}')
            # print(f'Way: {self.Way.shape}')
            # print(f'Hidden: {hidden.shape}')
            input_tanh = np.dot(self.Wax, input_x) +  np.dot(self.Waa , hidden)+ self.b
            hidden = layer_tanh.forward(input_tanh)
            self.hidden_list.append(hidden)
            
            
            input_softmax = np.dot(self.Way, hidden) + self.by
            y_pred = self.softmax.forward(input_softmax)
            self.y_preds.append(y_pred)

        return self.y_preds
    
    def loss(self, Y):

        self.Y = Y
        self.layers_loss = [CrossEntropyLoss() for y in self.Y]
        cost = 0

        for y_pred, y, layer in zip(self.y_preds, self.Y, self.layers_loss):
            cost+= layer.forward(y_pred, y)
        
        return cost


    
    def backward(self):

        gradients = self.__define_gradients()
        self.dWax, self.dWaa, self.dWya, self.db, self.dby, dhidden_next = gradients

        for index, layer_loss in reversed(list(enumerate(self.layers_loss))):
            dy = layer_loss.backward()

            hidden = self.hidden_list[index + 1]
            hidden_prev = self.hidden_list[index]

            #Gradients Y
            self.dWya += np.dot(dy, hidden.T)
            # print(f'dWya: {type(self.dWya)}')
            self.dby += dy
            dhidden = np.dot(self.Way.T , dy) + dhidden_next

            #Gardients A
            dtanh = self.layers_tanh[index].backward(dhidden)
            # print(f'dtanH: {type(dtanh)}')
            self.db += dtanh
            self.dWax += np.dot(dtanh, self.input_X[index].T)
            self.dWaa += np.dot(dtanh, hidden_prev.T)
            dhidden_next = np.dot(self.Waa.T, dtanh)

            

    def clip(self, clip_value):

        for gradient in [self.dWax, self.dWaa, self.dWya, self.db, self.dby]:
            np.clip(gradient, -clip_value, clip_value, out=gradient)

    
    def optimise(self, method):

        weights = [self.Way, self.Wax, self.Waa, self.by, self.b]
        gradients = [self.dWya, self.dWax, self.dWaa, self.dby, self.db]

        weights, self.oparams = method.optim(weights, gradients, self.oparams)
        self.Way, self.Wax, self.Waa, self.by, self.b = weights
        print(type(self.Wax))


    def __define_gradients(self):

        dWax = np.zeros_like(self.Wax)
        dWay = np.zeros_like(self.Way)
        dWaa = np.zeros_like(self.Waa)

        db = np.zeros_like(self.b)
        dby = np.zeros_like(self.by)

        da_next = np.zeros_like(self.hidden_list[0])

        return dWax, dWaa, dWay, db , dby, da_next
    
    def generate_names(
        self, index_to_character
    ):
       
        letter = None
        indexes = list(index_to_character.keys())

        letter_x = np.zeros((self.input_dim, 1))
        name = []

        # similar to forward propagation.
        layer_tanh = Tanh()
        hidden = np.zeros((self.hidden_dim , 1))
        # print(f'Letter X:{type(letter_x)}')
        # print(f'Wax: {type(self.Wax)}')
        # print(f'Waa: {type(self.Waa)}')
        # print(f'Way: {type(self.Way)}')
        # print(f'Hidden: {type(hidden)}')

        print(f'Letter X:{letter_x.shape}')
        print(f'Wax: {np.array(self.Wax).shape}')
        print(f'Waa: {np.array(self.Waa).shape}')
        print(f'Way: {np.array(self.Way).shape}')
        print(f'Hidden: {np.array(hidden).shape}')

        while letter != '\n' and len(name)<15:
            
            input_tanh = np.dot(self.Wax, letter_x) + np.dot(self.Waa, hidden) + self.b
            print(input_tanh)
            hidden = layer_tanh.forward(input_tanh)

            input_softmax = np.dot(self.Way, hidden) + self.by
            y_pred = self.softmax.forward(input_softmax)

            index = np.random.choice(indexes, p=y_pred.ravel())
            letter = index_to_character[index]

            name.append(letter)

            letter_x = np.zeros((self.input_dim, 1))
            letter_x[index] = 1

        return "".join(name)

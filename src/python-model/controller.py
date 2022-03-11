import json
from flask import Flask, render_template
import torch.nn.functional as F
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import torch.nn
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import MinMaxScaler, label_binarize
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math
from scipy.interpolate import make_interp_spline, interp1d
import seaborn as sns


controller = Flask(__name__)

# API receiver function
@controller.route('/pythonScript/<input_string>', methods=['POST'])
def pythonScriptFunction(input_string):
    print("CHECK POINT 1")
    coord = json.loads(input_string)
    print(coord)
    long = coord['long']
    lat = coord['lat']
    result = str(round(demo(lat, long)))
    return result


if __name__ == "__main__":
    controller.run(debug=True)


# ----- From here we are in our Machine Learning Function and call DEMO ----#

# Parameters used in model training (for hyperparameter search)
class Parameters:
    def __init__(self, input_dims=19, output_dims=1, epochs=1000,
                 learning_rate=0.1, batch_size=100, neurons=5,
                 loss_function=nn.MSELoss()):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neurons = neurons
        self.loss_function = loss_function


# Neural network class (leveraging PyTorch library)
class Regressor(nn.Module):
    transformer_X = None
    transformer_Y = None
    epochs = None

    def __init__(self, x, nb_epoch=1000, input_parameters=Parameters()):
        super(Regressor, self).__init__()
        # Training process
        self.nb_epoch = nb_epoch
        self.learning_rate = input_parameters.learning_rate
        self.batch_size = input_parameters.batch_size
        self.loss_function = nn.MSELoss()

        # Model configuration
        self.neurons = input_parameters.neurons
        self.input_layer = nn.Linear(input_parameters.input_dims, self.neurons)
        self.hidden_layer1 = nn.Linear(self.neurons, self.neurons)
        self.hidden_layer2 = nn.Linear(self.neurons, self.neurons)
        self.output_layer = nn.Linear(self.neurons,
                                      input_parameters.output_dims)
        self.optimizer = torch.optim.Adagrad(self.parameters(),
                                             lr=self.learning_rate)
        return

    # Function for data normalization (data has been pre-cleansed)
    def _preprocessor(self, x, y=None, training=False):

        # Storing data normalization ranges
        if training:
            self.transformer_X = MinMaxScaler(feature_range=(1, 100)).fit(x)
            if isinstance(y, pd.DataFrame):
                self.transformer_Y = MinMaxScaler(feature_range=(1, 100)).fit(y)


        # Applying data normalization
        if self.transformer_X is not None:
            x = pd.DataFrame(self.transformer_X.transform(x), index=x.index,
                             columns=x.columns)
            x_tensor = torch.tensor(x.values, dtype=torch.float32)
        else:
            raise "No normalization exists for input variables"
        if isinstance(y, pd.DataFrame):
            if self.transformer_Y is not None:
                y = pd.DataFrame(self.transformer_Y.transform(y), index=y.index,
                                 columns=y.columns)
                y_tensor = torch.tensor(y.values, dtype=torch.float32)
            else:
                raise "No normalization exists for housing prices"

        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)

     # Function for model training with the specified parameters
    def fit(self, x, y):
        # Data handling
        X, Y = self._preprocessor(x, y=y, training=True)
        train_dataset = TensorDataset(X, Y)
        train_dataloader = DataLoader(train_dataset, self.batch_size)
        self.train()

        # Training loop
        for epoch in range(self.nb_epoch):
            loss_curve = []
            for data, price in train_dataloader:
                self.optimizer.zero_grad()
                y_prediction = self.forward(data)
                loss = self.loss_function(y_prediction, price)
                loss.backward()
                self.optimizer.step()
                loss_curve += [loss.item()]
            if (epoch + 1) % 100 == 0:
                print("Epoch: " + str(epoch + 1) + "; Loss: "
                      + str(np.array(loss_curve).mean()))
        return self

    # Function for making model prediction (Variant: Unprocessed data)
    def predict(self, x):
        if not torch.is_tensor(x):
            x, _ = self._preprocessor(x, training=False)
        y = self.forward(x)
        y = y.detach().numpy()
        y = self.transformer_Y.inverse_transform(y)
        return y

    # Function for making model prediction (Variant: Preprocessed data)
    def forward(self, x):
        """
        This function illustrates the setup of the neural network, incl.:
            - Two hidden layers
            - Use of ReLu activation function - ReLu gives linear output for
                positive values, 0 for negative values. Since power output
                cannot be negative, this fits the model well.
        """

        y = self.input_layer(x)
        y = self.hidden_layer1(y)
        y = torch.relu(y)
        y = self.hidden_layer2(y)
        y = torch.relu(y)
        y = self.output_layer(y)
        y = F.relu(y)
        return y

    # Function to determine the 'loss' between predicted and actual output
    def score(self, x, y):
        # Note: Takes loss function as specified in regressor
        self.eval()
        X, Y = self._preprocessor(x, y=y, training=False)
        y_hat = self.predict(X)
        Y = self.transformer_Y.inverse_transform(Y)
        loss = self.loss_function(torch.tensor(Y), torch.tensor(y_hat))
        return math.sqrt(loss)

    # Function to specifically calculate the mean squared error of prediction
    def mse(self, x, y):
        self.eval()
        X, Y = self._preprocessor(x, y=y, training=False)
        y_hat = self.predict(X)
        Y = self.transformer_Y.inverse_transform(Y)
        loss = self.loss_function(torch.tensor(Y), torch.tensor(y_hat))
        return math.sqrt(loss)

    # Function to calculate R² for time series prediction
    def r2(self, x, y):
        self.eval()
        X, Y = self._preprocessor(x, y=y, training=False)
        y_hat = self.predict(X)
        Y = self.transformer_Y.inverse_transform(Y)
        return r2_score(Y, y_hat)


# Function to save the trained neural network to a .pickle file
def save_regressor(trained_model):
    with open('solar.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in solar.pickle\n")


# Function to load the pre-trained neural network
def load_regressor():
    with open('solar.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in solar.pickle\n")
    return trained_model

def RegressorHyperParameterSearch(x_train, y_train, x_val, y_val):
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyperparameters.
    """

    base_error = -1
    for epoch in range(1000, 2001, 1000):
        for batch_size in range(100, 201, 100):
            for learning_rate in np.arange(0.15, 0.16, 0.1):
                for neurons in range(6, 20, 2):
                    parameters = Parameters(epochs=epoch, learning_rate=learning_rate, batch_size=batch_size,
                                            neurons=neurons)
                    reg = Regressor(x_train, nb_epoch=parameters.epochs, input_parameters=parameters)
                    reg.fit(x_train, y_train)
                    error = reg.r2(x_val, y_val)
                    print("Epoch: " + str(reg.nb_epoch) + "; LR: " + str(
                        reg.learning_rate) + "; Batch size: " + str(reg.batch_size) + "; Neurons: " + str(
                        reg.neurons) + "; R²: " + str(error))
                    if (error > base_error) or (base_error == -1):
                        best_parameters = parameters
                        base_error = error
                        print("This is currently the best model")
                    # if error > 0.8:
                    # save_regressor(reg)
                    # return best_parameters
    return best_parameters

# Function to perform search for best hyperparameters
def RegressorHyperParameterSearch(x_train, y_train, x_val, y_val):
    base_error = -1
    for epoch in range(1000, 2001, 1000):
        for batch_size in range(100, 201, 100):
            for learning_rate in np.arange(0.15, 0.16, 0.1):
                for neurons in range(6, 20, 2):
                    parameters = Parameters(epochs=epoch,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size,
                                            neurons=neurons)
                    reg = Regressor(x_train, nb_epoch=parameters.epochs,
                                    input_parameters=parameters)
                    reg.fit(x_train, y_train)
                    error = reg.r2(x_val, y_val)
                    print("Epoch: " + str(reg.nb_epoch) + "; LR: " + str(
                        reg.learning_rate) + "; Batch size: " +
                          str(reg.batch_size) + "; Neurons: " +
                          str(reg.neurons) + "; R²: " + str(error))
                    if (error > base_error) or (base_error == -1):
                        best_parameters = parameters
                        base_error = error
                        print("This is currently the best model")
    return best_parameters


# Function to find the right model parameters and initiate training
def train():
    # Read in data
    output_label = "SolarEnergy"
    training_data = pd.read_csv("weather_and_power_sites_noElm.csv")
    test_location = pd.read_csv("weather_and_power_sites_ElmOnly.csv")
    """
    Note: We have pre-split the data into a training set ('training_data') 
    and a test set ('test_location'). The test set encompasses the weather
    readings of a single site over the course of a year. Whereas we recognize
    that the split between test and training data is usually performed randomly,
    the performance between random seeding and site-specific seeding (as 
    implemented) was observed to be very similar. Furthermore, the site-specific
    seeding allows us to run a demonstration of the model on the weather readings
    of a single site over a year.
    
    The site-specific split results in a training-test split of 79% to 21%, which
    is in line with best practice. 
    """

    # Further splitting of training data into pure training and validation set
    data_train, data_val = train_test_split(training_data, test_size=0.2,
                                            random_state=1203)
    x_train = data_train.loc[:, training_data.columns != output_label]
    y_train = data_train.loc[:, [output_label]]
    x_val = data_val.loc[:, training_data.columns != output_label]
    y_val = data_val.loc[:, [output_label]]
    x_test = test_location.loc[:, test_location.columns != output_label]
    y_test = test_location.loc[:, [output_label]]

    # Hyperparameter search
    parameters = RegressorHyperParameterSearch(x_train, y_train, x_val, y_val)
    # parameters = Parameters(epochs=1000, batch_size=100, learning_rate=0.15, neurons=10)
    regressor = Regressor(x_train, nb_epoch=parameters.epochs,
                          input_parameters=parameters)

    # For final model training, merging pure training and validation sets
    x_train = pd.concat([x_train, x_val])
    y_train = pd.concat([y_train, y_val])
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    mse = regressor.mse(x_test, y_test)
    r2 = regressor.r2(x_test, y_test)
    print("Regressor RMSE: " + str(mse))
    print("R²: " + str(r2))

# Function to demonstrate models prediction capabilities (used for website)
def demo(lat, long):
    # Data preparation
    lat = float(lat)
    long = float(long)
    output_label = "SolarEnergy"
    data_predict = pd.read_csv("weather_and_power_sites_ElmOnly.csv")
    data_predict = data_predict[data_predict.lat == lat]
    data_predict = data_predict.drop(columns=['lat'])
    data_predict = data_predict[data_predict.long == long]
    data_predict = data_predict.drop(columns=['long'])
    x = data_predict.loc[:, data_predict.columns != output_label]
    y = data_predict.loc[:, [output_label]]

    # Regressor run
    reg = load_regressor()
    X, Y = reg._preprocessor(x, y=y, training=False)
    y_hat = reg.predict(X)
    mse = reg.mse(x, y)
    r2 = reg.r2(x, y)

    # Analysis of kWH input on per-week basis
    conversion = 0.011622222  # Note: Converts langleys (output of model) to kWh
    kW_installed = 4  # Note: Capacity of unit installed, 4kW is standard issue
    kWh = pd.DataFrame(y_hat * conversion * kW_installed, columns=['kWh'])
    date_time = pd.to_datetime(x[['Year', 'Month', 'Day', 'Hour']])
    kWh.insert(0, "Week", date_time.dt.week)
    kWh_grouped = kWh.groupby(['Week']).sum()

    # Plotting graph of output (weekly basis)
    sns.set_style("whitegrid")
    blue, = sns.color_palette("muted", 1)
    x = kWh_grouped.index.to_numpy()
    y = kWh_grouped['kWh'].to_numpy()
    cubic_interploation_model = interp1d(x, y, kind="cubic")
    X_ = np.linspace(x.min(), x.max(), 5000)
    Y_ = cubic_interploation_model(X_)
    #plt.plot(X_, Y_)
    #ax = plt.subplot(111)
    #l = ax.fill_between(X_, Y_)
    #plt.title("Predicted Power Generation by Week \n (Neural Network Output)")
    #plt.xlabel("Week")
    #plt.ylabel("kWh")
    #plt.savefig('graph.png')
    #plt.show()

    total_power = np.sum(y)
    print('Annualized power generation: ' + str(total_power))
    print("Regressor RMSE: " + str(mse))
    print("R²: " + str(r2))
    return total_power


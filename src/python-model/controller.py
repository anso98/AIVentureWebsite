import json
from MLmodel import *
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


# FROM HERE THE NEURALNETWORK CODE 
# WE ARE CALLING DEMO(lat, long)

class Parameters:
    def __init__(self, input_dims=19, output_dims=1, epochs=1000, learning_rate=0.1, batch_size=100,
                 neurons=5, loss_function=nn.MSELoss()):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neurons = neurons
        self.loss_function = loss_function


class Regressor(nn.Module):
    transformer_X = None
    transformer_Y = None
    nb_epoch = None

    def __init__(self, x, nb_epoch=1000, input_parameters=Parameters()):
        super(Regressor, self).__init__()
        # Training setup
        self.nb_epoch = nb_epoch
        self.learning_rate = input_parameters.learning_rate
        self.batch_size = input_parameters.batch_size
        self.loss_function = nn.MSELoss()

        # Network setup
        self.neurons = input_parameters.neurons
        self.input_layer = nn.Linear(input_parameters.input_dims, self.neurons)
        self.hidden_layer1 = nn.Linear(self.neurons, self.neurons)
        self.hidden_layer2 = nn.Linear(self.neurons, self.neurons)
        self.output_layer = nn.Linear(self.neurons, input_parameters.output_dims)
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        # X, _ = self._preprocessor(x, training=True)
        return

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        # Store parameters for pre-processing
        if training:
            self.transformer_X = MinMaxScaler(feature_range=(1, 100)).fit(x)
            if isinstance(y, pd.DataFrame):
                self.transformer_Y = MinMaxScaler(feature_range=(1, 100)).fit(y)

        # Apply preprocessing to X and Y
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

        # Return preprocessed x and y, return None for y if it was None
        return x_tensor, (y_tensor if isinstance(y, pd.DataFrame) else None)

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.
        """

        X, Y = self._preprocessor(x, y=y, training=True)
        train_dataset = TensorDataset(X, Y)
        train_dataloader = DataLoader(train_dataset, self.batch_size)
        self.train()

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
                print("Epoch: " + str(epoch + 1) + "; Loss: " + str(np.array(loss_curve).mean()))
        return self

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).
        """
        if not torch.is_tensor(x):
            x, _ = self._preprocessor(x, training=False)
        y = self.forward(x)
        y = y.detach().numpy()
        y = self.transformer_Y.inverse_transform(y)
        return y

    def forward(self, x):
        y = self.input_layer(x)
        y = self.hidden_layer1(y)
        y = torch.relu(y)
        y = self.hidden_layer2(y)
        y = torch.relu(y)
        y = self.output_layer(y)
        y = F.relu(y)
        return y

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        self.eval()
        X, Y = self._preprocessor(x, y=y, training=False)
        y_hat = self.predict(X)
        Y = self.transformer_Y.inverse_transform(Y)
        loss = self.loss_function(torch.tensor(Y), torch.tensor(y_hat))
        return math.sqrt(loss)

    def mse(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        self.eval()
        X, Y = self._preprocessor(x, y=y, training=False)
        y_hat = self.predict(X)
        Y = self.transformer_Y.inverse_transform(Y)
        loss = self.loss_function(torch.tensor(Y), torch.tensor(y_hat))
        return math.sqrt(loss)

    def r2(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        self.eval()
        X, Y = self._preprocessor(x, y=y, training=False)
        y_hat = self.predict(X)
        Y = self.transformer_Y.inverse_transform(Y)
        return r2_score(Y, y_hat)


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('solar.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in solar.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('./solar.pickle', 'rb') as target:
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


def train():
    # Read in data
    output_label = "SolarEnergy"
    data_noElm = pd.read_csv("weather_and_power_sites_noElm.csv")
    data_elmOnly = pd.read_csv("weather_and_power_sites_ElmOnly.csv")

    # Splitting input and output
    data_train, data_val = train_test_split(data_noElm, test_size=0.2, random_state=1203)
    x_train = data_train.loc[:, data_noElm.columns != output_label]
    y_train = data_train.loc[:, [output_label]]
    x_val = data_val.loc[:, data_noElm.columns != output_label]
    y_val = data_val.loc[:, [output_label]]
    x_test = data_elmOnly.loc[:, data_elmOnly.columns != output_label]
    y_test = data_elmOnly.loc[:, [output_label]]

    # Hyperparameter search
    #parameters = RegressorHyperParameterSearch(x_train, y_train, x_val, y_val)
    parameters = Parameters(epochs=1000, batch_size=100, learning_rate=0.15, neurons=10)
    regressor = Regressor(x_train, nb_epoch=parameters.epochs, input_parameters=parameters)

    # Merge train and val back together for final training
    x_train = pd.concat([x_train, x_val])
    y_train = pd.concat([y_train, y_val])
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    mse = regressor.mse(x_test, y_test)
    r2 = regressor.r2(x_test, y_test)
    print("Regressor RMSE: " + str(mse))
    print("R²: " + str(r2))


def demo(lat, long):
    # Data preparation
    output_label = "SolarEnergy"
    lat = float(lat)
    long = float(long)
    #print("lat" + lat)
    #print("long" + long)
    print(type(lat))
    print(type(long))
    data_predict = pd.read_csv("weather_and_power_sites_ElmOnly.csv")
    print(data_predict)
    print(type(data_predict.lat.values[0]))
    data_predict = data_predict[data_predict.lat == lat]
    print(data_predict)
    data_predict = data_predict.drop(columns=['lat'])
    data_predict = data_predict[data_predict.long == long]
    print(data_predict)
    data_predict = data_predict.drop(columns=['long'])
    x = data_predict.loc[:, data_predict.columns != output_label]
    y = data_predict.loc[:, [output_label]]

    # Regressor run
    reg = load_regressor()
    X, Y = reg._preprocessor(x, y=y, training=False)
    y_hat = reg.predict(X)
    mse = reg.mse(x, y)
    r2 = reg.r2(x, y)

    # Analyze kWh output
    conversion = 0.011622222
    kW_installed = 4
    full_year_scale = 1  # 365 / 300 * 0.9
    kWh = pd.DataFrame(y_hat * conversion * kW_installed, columns=['kWh'])
    actuals = pd.DataFrame(Y.detach().numpy() * conversion * kW_installed, columns=['kWh'])
    date_time = pd.to_datetime(x[['Year', 'Month', 'Day', 'Hour']])
    kWh.insert(0, "Week", date_time.dt.week)
    kWh_grouped = kWh.groupby(['Week']).sum()
    actuals.insert(0, "Week", date_time.dt.week)
    actuals_grouped = actuals.groupby(['Week']).sum()

    # Calculate st. deviation
    errors = np.array(abs(actuals_grouped['kWh'].to_numpy() - kWh_grouped['kWh'].to_numpy()))
    stdev = np.std(errors)

    # Plot graph
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

    total_power = np.sum(y) * full_year_scale
    stddev_annualized = stdev * math.sqrt(55)
    z = 1.96
    lower_bound = total_power - z * stddev_annualized
    upper_bound = total_power + z * stddev_annualized

    print('Annualized power generation: ' + str(total_power))
    print('95% Confidence interval: ' + str(round(lower_bound)) + ' - ' + str(round(upper_bound)))
    print("Regressor RMSE: " + str(mse))
    print("R²: " + str(r2))
    print("St.Dev: " + str(stdev * math.sqrt(44)))
    return total_power


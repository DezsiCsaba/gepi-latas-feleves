
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

class linearRegressionNN():
    def __init__(self, network:nn.Sequential):
        self.model = network

    def setTrainTestDatas(self, X_train_raw, y_train, X_test_raw, y_test):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_raw)
        x_train =self.scaler.transform(X_train_raw)
        x_test = self.scaler.transform(X_test_raw)

        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        self.X_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    def fit(self, epochs, batch_size, disableTrainBars, loss_fn, opt_fn):
        batch_start = torch.arange(0, len(self.X_train), batch_size)

        self.best_mse = np.inf   # init to infinity
        self.best_weights = None
        self.history = []
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn

        for epoch in range(epochs):
            self.model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=disableTrainBars) as bar:
                bar.set_description(f"Epoch {epoch+1}")
                for start in bar:
                    # take a batch
                    X_batch = self.X_train[start : start+batch_size]
                    y_batch = self.y_train[start : start+batch_size]
                    # forward pass
                    y_pred = self.model(X_batch)
                    loss = self.loss_fn(y_pred, y_batch)
                    # backward pass
                    self.opt_fn.zero_grad()
                    loss.backward()
                    # update weights
                    self.opt_fn.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            mse = float(loss)
            self.history.append(mse)
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_weights = copy.deepcopy(self.model.state_dict())
                self.modelState = self.model.load_state_dict(self.best_weights)
                torch.save({
                    'model_state_dict': self.modelState,
                    }, 'models/bestLinRegNN.pt')
            
            if (disableTrainBars==True):
                print(f'{epoch+1}.epoch -> MSE: {round(mse, 3)}')
                print("\t >>> our best MSE so far: %.2f" % self.best_mse)

        

    def plotTrainingHistory(self):
        print("MSE: %.4f" % self.best_mse)
        print("RMSE: %.4f" % np.sqrt(self.best_mse))
        plt.plot(self.history)
        plt.show()

    
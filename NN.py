
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

import joblib
import pickle
import json
#import sklearn_json as skljson

class neuralNetwork():
    def __init__(self, network:nn.Sequential = None):
        self.model = network

    def setTrainTestDatas(self, X_train_raw, y_train, X_test_raw, y_test):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_raw)

        print('before scaling:', X_train_raw[0:1])
        x_train =self.scaler.transform(X_train_raw)
        print('after scaling:', x_train[0:1])
        print('x_train shape before:',np.shape(X_train_raw[0:1]))
        print('x_train shape after:',np.shape(x_train[0:1]))

        x_test = self.scaler.transform(X_test_raw)

        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        self.X_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    def fit(self, epochs, batch_size, disableTrainBars, loss_fn, opt_fn, iter, iterHistory, saveAs):
        batch_start = torch.arange(0, len(self.X_train), batch_size)

        self.best_MAE = np.inf   # init to infinity
        self.global_best_MAE = 0.002
        self.best_weights = None
        self.history = iterHistory
        self.loss_fn = loss_fn
        self.opt_fn = opt_fn

        for epoch in range(epochs):
            self.model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=disableTrainBars) as bar:
                bar.set_description(f"Epoch {epoch+1}, iteration {iter+1}")
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
                    bar.set_postfix(MAE=float(loss))
            MAE = float(loss)
            self.history.append(MAE)
            if MAE < self.best_MAE:
                self.best_MAE = MAE
                self.best_weights = copy.deepcopy(self.model.state_dict())

                if (self.best_MAE < self.global_best_MAE):
                    self.global_best_MAE = self.best_MAE
                    #torch.save(self.model.state_dict(), 'models/LinRegNN_best.pt')
                    torch.save(self.model.state_dict(), f'models/{saveAs}.pt')
                    #print('\tNEW BEST MODEL! :)')
                else: 
                    torch.save(self.model.state_dict(), f'models/{saveAs}.pt')
            
            if (disableTrainBars==True):
                print(f'{epoch+1}.epoch -> MAE: {round(MAE, 3)}')
                print("\t >>> our best MAE so far: %.2f" % self.best_MAE)
        
        finished = torch.load('models/LinRegNN.pt')
        self.model.load_state_dict(finished)
        #skljson.to_json(self.scaler, 'jsonScaler.json')
        #self.saveScaler()
        
    def plotTrainingHistory(self):
        if (self.best_MAE < self.global_best_MAE):
            print('new global best')

        print("MAE: %.4f" % self.best_MAE)
        print("RMAE: %.4f" % np.sqrt(self.best_MAE))
        plt.plot(self.history)
        plt.show()

    def compareWithBest(self, X_test_raw, y_test, sample_size, defaultNetwork, bestPath, modelPath):
        best = torch.load(f'models/{bestPath}.pt')
        loaded = torch.load(f'models/{modelPath}.pt')

        bestReg = neuralNetwork(defaultNetwork)
        bestReg.model.load_state_dict(best)
        self.model.load_state_dict(loaded)

        res=[]
        res2=[]
        expecteds=[]
        infers1=[]
        infers2=[]
        base=[]
        with torch.no_grad():
            for i in range(sample_size):
                X_sample = X_test_raw[i: i+1]
                X_sample = self.scaler.transform(X_sample)
                X_sample = torch.tensor(X_sample, dtype=torch.float32)
                
                y_pred = self.model(X_sample)
                y_pred2 = bestReg.model(X_sample)

                result = round(y_pred[0].numpy()[0]*1.0, 2)
                result2 = round(y_pred2[0].numpy()[0]*1.0, 2)
                expected = round(y_test.to_numpy()[i]*1.0, 2)
                inference = round(result-expected, 2)
                inference2 = round(result2-expected, 2)

                res.append(result)
                res2.append(result2)
                expecteds.append(expected)
                infers1.append(inference)
                infers2.append(inference2)
                base.append(0)
                #print(f'result = {result}, bestResult = {result2}, expected = {expected}')

        plt.figure(figsize=(15, 3))
        #plt.plot(expecteds, 'g')
        plt.plot(infers1, 'b', linewidth=1)
        plt.plot(infers2, 'r')
        plt.plot(base, 'g', linewidth=3)

        plt.show()

    def evaluate(self, X_test_raw, y_test, sample_size=None):
        print('tensor done')
        res=[]
        exp=[]
        inf=[]
        with torch.no_grad():
            for i in range(len(X_test_raw)):
                X_sample = X_test_raw[i: i+1]
                X_sample = self.scaler.transform(X_sample)
                X_sample = torch.tensor(X_sample, dtype=torch.float32)
                
                y_pred = self.model(X_sample)

                result = round(y_pred[0].numpy()[0]*1.0, 2)
                expected = round(y_test.to_numpy()[i]*1.0, 2)
                inference = round(result-expected, 2)

                res.append(result)
                exp.append(expected)
                inf.append(np.abs(inference))

        plt.figure(figsize=(15, 3))
        plt.plot(res, 'g', linewidth=1)
        plt.plot(exp, 'b', linewidth=1)
        plt.plot(inf, 'r', linewidth=2)
        plt.show()
        print(f'AVG MAE:{np.mean(inf)}')
        print(f'MAX MAE:{np.max(inf)}')
        print(f'MIN MAE:{np.min(inf)}')


    def saveScaler(self):
        dict = self.scaler.__dict__
        import json
        for k, v in dict.items():
            if isinstance(v, np.int64):
                dict[k] = int(v)
            if isinstance(v, np.ndarray):
                if  k[-1:] == '_':
                    dict[k] = v.tolist()

        with open('service/sk_scaler.json', 'w', encoding='utf-8') as f:
            json.dump(dict, f, ensure_ascii=False, indent=4)


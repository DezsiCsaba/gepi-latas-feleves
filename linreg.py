import torch.nn as nn
import torch.optim as optim
from NN import neuralNetwork
import torch.onnx


class LinearRegressionNN():
    def __init__(self):
        self.activation_fn = nn.ReLU
        self.network = nn.Sequential(
            nn.Linear(6, 10),
            self.activation_fn(),
            nn.Linear(10, 10),
            self.activation_fn(),
            nn.Linear(10, 1),
        )        
        self.loss_fn = nn.L1Loss() #MAE
        self.opt_fn = optim.Adam(self.network.parameters(), lr=0.0015)
    
    def createLinRegModel(self):
        self.linreg = neuralNetwork(network = self.network)
        
    def trainNetwork(self, X_train_raw, y_train, X_test_raw, y_test, epochs = 150, batch_num=10, iteration=1):
        iterHistory = []
        for i in range(iteration):
            self.linreg.setTrainTestDatas(X_train_raw, y_train, X_test_raw, y_test)
            self.linreg.fit(
                epochs=epochs,
                batch_size=batch_num,
                disableTrainBars=False,
                loss_fn=self.loss_fn,
                opt_fn=self.opt_fn,
                iter=i,
                iterHistory=iterHistory,
                saveAs='LinRegNN'
            )
            iterHistory = self.linreg.history

    def plotTrainingHistory(self):
        self.linreg.plotTrainingHistory()

    def compareWithBestLinregModel(self, X_test_raw, y_test, sample_size=70):
        default = nn.Sequential(
            nn.Linear(6, 10),
            self.activation_fn(),
            nn.Linear(10, 10),
            self.activation_fn(),
            nn.Linear(10, 1),
        )
        self.linreg.compareWithBest(
            X_test_raw,
            y_test,
            sample_size,
            default,
            bestPath='LinRegNN_best',
            modelPath='LinRegNN'
        )

    def evaluateModel(self, X_test_raw, y_test):
        self.linreg.evaluate(X_test_raw, y_test, 10)

    def export_to_ONNX_model(self):
        '''If model is not a torch.jit.ScriptModule nor
            a torch.jit.ScriptFunction, this runs model once in order to
            convert it to a TorchScript graph to be exported''' #-> ezért adunk be neki inputot, hogy tudjon futni ha kell neki

        self.linreg.model.eval()
        X_test = self.linreg.X_test

        torch.onnx.export(
            self.linreg.model,
            X_test[0],
            "models/linreg_OnnxModel.onnx", # where to save the model
            verbose = True

            #export_params = True, # store the trained parameter weights inside the model file
            #opset_version = 18 #onnx version to export to
        )
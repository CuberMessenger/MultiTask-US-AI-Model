import os
import time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.utils.data as TorchData
import DataUtils.ReadData as ReadData
import DataUtils.PrintAndPlot as PrintAndPlot
from DataUtils.CustomedDataset import TensorDatasetWithTransform, UltrasoundDataTransform

class MachineLearningModel:
    def __init__(self, earlyStoppingPatience, learnRate, batchSize):
        self.EarlyStoppingPatience = earlyStoppingPatience
        self.LearnRate = learnRate
        self.BatchSize = batchSize
        self.Epsilon = 1e-6
        self.BestStateDict = None

        self.SetupSeed(2223)

    def SetupSeed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def LoadStateDictionary(self, stateDictionary):
        self.Net.load_state_dict(stateDictionary, strict = True)

    def LoadSet(self, set, fold, transform = None):
        setDataIndex = fold[set + "SetDataIndex"]
        setLabel = torch.from_numpy(fold[set + "SetLabel"]).long()
        setDataset = TensorDatasetWithTransform([setLabel],
                                                fold["DataPath"],
                                                setDataIndex,
                                                transform = transform)
        setLoader = TorchData.DataLoader(setDataset, batch_size = self.BatchSize, num_workers = 8, shuffle = False if set == "Test" else True)
        return setDataIndex, setLabel, setLoader

    def LoadData(self, fold):
        self.TrainDataIndex, self.TrainLabel, self.TrainLoader = self.LoadSet("Train", fold, UltrasoundDataTransform)
        self.ValidationDataIndex, self.ValidationLabel, self.ValidationLoader = self.LoadSet("Validation", fold)

def EvaluateMachineLearningModel(modelClass,\
                                 saveFolderPath, trainPaths, testPaths,\
                                 earlyStoppingPatience = 20, learnRate = 0.0001, batchSize = 64,\
                                 name = None):
    folds = ReadData.ReadFolds(trainPaths)
    testFold = ReadData.ReadFolds(testPaths)

    saveFolderPath = os.path.join(saveFolderPath, modelClass.__name__ + ("" if name is None else name) + time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))
    os.mkdir(saveFolderPath)

    validationAnswers = []
    testAnswers = []
    try:
        for i in range(folds.NumOfFold):
            model = modelClass(earlyStoppingPatience = earlyStoppingPatience, learnRate = learnRate, batchSize = batchSize)
            model.Net = nn.DataParallel(model.Net, device_ids = [0, 1, 3] if os.name == "posix" else [0])
            model.Net = model.Net.cuda()
            model.LossFunction = model.LossFunction.cuda()
            model.LoadData(folds.NextFold())
            validationAnswer = model.Train()
            validationAnswers.append(validationAnswer)

            model.TestDataIndex, model.TestLabel, model.TestLoader =  model.LoadSet("Test", testFold.GetWholeAsTest())
            testAnswer, _ = model.Evaluate(model.TestLoader, model.BestStateDict)
            testAnswers.append(testAnswer)
            torch.save(model.BestStateDict, os.path.join(saveFolderPath, "Fold%dWeights.pkl" % i))
    except Exception:
        os.rmdir(saveFolderPath)
        raise

    PrintAndPlot.ClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath)
    shutil.copy("log.log", os.path.join(saveFolderPath, "log.log"))
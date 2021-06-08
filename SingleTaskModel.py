import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from termcolor import colored
from DataUtils.PrintAndPlot import SingleTaskClassificationAnswer
from NeutralNetworkUtils.InceptionResNetV2 import InceptionResNetV2
from StatisticsUtils import ClassificationMetrics, BinaryClassificationMetric
from MachineLearningModel import MachineLearningModel, EvaluateMachineLearningModel

class SingleTaskModel(MachineLearningModel):
    def __init__(self, earlyStoppingPatience, learnRate, batchSize):
        super().__init__(earlyStoppingPatience, learnRate, batchSize)
        self.Net = InceptionResNetV2(numOfClasses = 2)
        self.LossFunction = nn.CrossEntropyLoss()

    def Train(self):
        epoch = 0
        patience = self.EarlyStoppingPatience
        optimizer = torch.optim.Adam(self.Net.parameters(), lr = self.LearnRate)
        numOfInstance = len(self.TrainLabel)
        minLoss = float(0x7FFFFFFF)
        bestValidationAnswer = None

        trainLosses = []
        validationLosses = []

        while patience > 0:
            self.Net.train()
            epoch += 1
            runningLoss = 0.0
            for batchImage, batchLabel, _ in self.TrainLoader:
                batchImage = batchImage.float().cuda()
                batchLabel = batchLabel[:, 0].long().cuda()
                optimizer.zero_grad()

                outputClass = self.Net.forward(batchImage)
                loss = self.LossFunction(outputClass, batchLabel)

                loss = loss.mean()
                runningLoss += loss.item()
                loss.backward()
                optimizer.step()

            self.Net.eval()
            trainLoss = (runningLoss * self.BatchSize) / numOfInstance
            validationAnswer, validationLoss = self.Evaluate(self.ValidationLoader, None)

            trainLosses.append(trainLoss)
            validationLosses.append(validationLoss)

            print("Epoch %d:\ntrainLoss -> %.3f, valLoss -> %.3f" % (epoch, trainLoss, validationLoss))
            print("Accuracy -> %f" % validationAnswer.Accuracy, end = ", ")
            print("Recall -> %f" % validationAnswer.Recall, end = ", ")
            print("Precision -> %f" % validationAnswer.Precision, end = ", ")
            print("Specificity -> %f" % validationAnswer.Specificity)

            if minLoss > validationLoss:
                patience = self.EarlyStoppingPatience
                minLoss = validationLoss
                bestValidationAnswer = validationAnswer
                self.BestStateDict = copy.deepcopy(self.Net.state_dict())
                print(colored("Better!!!!!!!!!!!!!!!!!!!!!!!!!!!", "green"))
            else:
                patience -= 1
                print(colored("Worse!!!!!!!!!!!!!!!!!!!!!!!!!!!", "red"))

        bestValidationAnswer.TrainLosses = trainLosses
        bestValidationAnswer.ValidationLosses = validationLosses
        return bestValidationAnswer

    def Evaluate(self, dataLoader, stateDictionary):
        self.Net.eval()
        answer = SingleTaskClassificationAnswer()
        if stateDictionary is not None:
            self.LoadStateDictionary(stateDictionary)
        with torch.no_grad():
            numOfInstance = len(dataLoader.dataset)
            runningLoss = 0.0
            TP = [0]
            FP = [0]
            TN = [0]
            FN = [0]

            for batchImage, batchLabel, batchDataIndex in dataLoader:
                batchImage = batchImage.float().cuda()
                batchLabel = batchLabel[:, 0].long().cuda()

                outputClass = self.Net.forward(batchImage)
                loss = self.LossFunction(outputClass, batchLabel)
                
                loss = loss.mean()
                runningLoss += loss.item()

                BinaryClassificationMetric(outputClass, batchLabel, TP, FP, TN, FN, 0)

                answer.Outputs = torch.cat((answer.Outputs, outputClass.softmax(dim = 1).cpu()), dim = 0)
                answer.Labels = torch.cat((answer.Labels, batchLabel.float().cpu()), dim = 0)
                answer.DataIndexes += batchDataIndex

            answer.Accuracy, answer.Recall, answer.Precision, answer.Specificity = ClassificationMetrics(TP[0], FP[0], TN[0], FN[0], self.Epsilon)
            loss = (runningLoss * self.BatchSize) / numOfInstance
            return answer, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--TrainFolderPath", help = "define the train data folder path", type = str)
    parser.add_argument("--TestFolderPath", help = "define the test data folder path", type = str)
    parser.add_argument("--TrainInfoPath", help = "define the train info path", type = str)
    parser.add_argument("--TestInfoPath", help = "define the test info path", type = str)
    parser.add_argument("--SaveFolderPath", help = "define the save folder path", type = str)
    parser.add_argument("--Name", help = "define the name", type = str)
    args = parser.parse_args()

    EvaluateMachineLearningModel(SingleTaskModel,\
                                 args.SaveFolderPath, (args.TrainFolderPath, args.TrainInfoPath), (args.TestFolderPath, args.TestInfoPath),\
                                 name = args.Name)
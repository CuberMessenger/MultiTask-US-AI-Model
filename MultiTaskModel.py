import os
import csv
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from termcolor import colored
from StatisticsUtils import CalculateAUC
from NeutralNetworkUtils.BlockUtils import MultiTaskBlock
from DataUtils.PrintAndPlot import MultiTaskClassificationAnswer
from MachineLearningModel import MachineLearningModel, EvaluateMachineLearningModel
from NeutralNetworkUtils.InceptionResNetV2 import InceptionResNetV2, MultiScaleBlock
from DataUtils.CustomedDataset import TensorDatasetWithTransform, UltrasoundDataTransform
from StatisticsUtils import ClassificationMetrics, BinaryClassificationMetric, MultipleClassificationMetric

class MultiTaskCrossEntropyLoss(nn.Module):
    def __init__(self, taskMask = range(1, 11)):
        super().__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.TaskMask = taskMask

    def forward(self, outputClass, targetClass, net = None):
        labelLoss = self.CrossEntropyLoss(outputClass[0], targetClass[:, 0])
        classificationLoss = 0 + labelLoss
        for i in self.TaskMask:
            classificationLoss += self.CrossEntropyLoss(outputClass[i], targetClass[:, i])
        return classificationLoss, labelLoss

class MultiTaskModel(MachineLearningModel):
    def __init__(self, earlyStoppingPatience, learnRate, batchSize):
        super().__init__(earlyStoppingPatience, learnRate, batchSize)
        self.Net = InceptionResNetV2(numOfClasses = 2)
        self.Net.LastLinear = MultiTaskBlock(1536, 512)
        self.Net.Convolution1A = MultiScaleBlock(inChannel = 1, outChannel = 32, stride = 2)
        self.Net.Convolution2A = MultiScaleBlock(inChannel = 32, outChannel = 32, stride = 1)
        self.Net.Convolution2B = MultiScaleBlock(inChannel = 32, outChannel = 64, stride = 1, padding = 1)
        self.LossFunction = MultiTaskCrossEntropyLoss()

    def Train(self):
        epoch = 0
        patience = self.EarlyStoppingPatience
        optimizer = torch.optim.Adam(self.Net.parameters(), lr = self.LearnRate)
        numOfInstance = len(self.TrainLabel)
        minLoss = float(0x7FFFFFFF)
        bestValidationAnswer = None

        trainLosses = []
        trainLabelLosses = []
        validationLosses = []
        validationLabelLosses = []

        while patience > 0:
            self.Net.train()
            epoch += 1
            runningLoss = 0.0
            labelLoss = 0.0
            for batchImage, batchLabel, _ in self.TrainLoader:
                batchImage = batchImage.float().cuda()
                batchLabel = batchLabel.long().cuda()
                optimizer.zero_grad()

                outputClass = self.Net.forward(batchImage)
                rawLoss = self.LossFunction(outputClass, batchLabel)

                loss, lLoss = rawLoss
                loss = loss.mean()
                lLoss = lLoss.mean()

                runningLoss += loss.item()
                labelLoss += lLoss.item()
                loss.backward()
                optimizer.step()

            self.Net.eval()
            trainLoss = (np.array([runningLoss, labelLoss]) * self.BatchSize) / numOfInstance
            validationAnswer, validationLoss = self.Evaluate(self.ValidationLoader, None)

            trainLosses.append(trainLoss[0])
            trainLabelLosses.append(trainLoss[1])
            validationLosses.append(validationLoss[0])
            validationLabelLosses.append(validationLoss[1])

            print("Epoch %d:\ntrainLoss -> %.5f, trainLabelLoss -> %.5f, valLoss -> %.5f, valLabelLoss -> %.5f" % (epoch, trainLoss[0], trainLoss[1], validationLoss[0], validationLoss[1]))
            print("\t\tAccuracy\tRecall\t\tPrecision\tSpecificity")
            taskNames = ["Label\t", "Composition", "Echogenicity", "Shape\t", "Margin\t", "Irregular", "Extension", "CometTail", "Macro\t", "Peripheral", "Punctate"]
            for i in range(11):
                print("%s\t%.5f\t\t%.5f\t\t%.5f\t\t%.5f" % (taskNames[i],
                                                            validationAnswer.Accuracy[i],
                                                            validationAnswer.Recall[i],
                                                            validationAnswer.Precision[i],
                                                            validationAnswer.Specificity[i]))
            if minLoss > validationLoss[0]:
                patience = self.EarlyStoppingPatience
                minLoss = validationLoss[0]
                bestValidationAnswer = validationAnswer
                self.BestStateDict = copy.deepcopy(self.Net.state_dict())
                print(colored("Better!!!!!!!!!!!!!!!!!!!!!!!!!!!", "green"))
            else:
                patience -= 1
                print(colored("Worse!!!!!!!!!!!!!!!!!!!!!!!!!!!", "red"))

        bestValidationAnswer.TrainLosses = trainLosses
        bestValidationAnswer.TrainLabelLosses = trainLabelLosses
        bestValidationAnswer.ValidationLosses = validationLosses
        bestValidationAnswer.ValidationLabelLosses = validationLabelLosses
        return bestValidationAnswer

    def Evaluate(self, dataLoader, stateDictionary):
        self.Net.eval()
        answer = MultiTaskClassificationAnswer()
        if stateDictionary is not None:
            self.LoadStateDictionary(stateDictionary)
        with torch.no_grad():
            numOfInstance = len(dataLoader.dataset)
            runningLoss = 0.0
            labelLoss = 0.0
            TP = [0, [0] * 4, [0] * 5, 0, 0, 0, 0, 0, 0, 0, 0]
            FP = [0, [0] * 4, [0] * 5, 0, 0, 0, 0, 0, 0, 0, 0]
            TN = [0, [0] * 4, [0] * 5, 0, 0, 0, 0, 0, 0, 0, 0]
            FN = [0, [0] * 4, [0] * 5, 0, 0, 0, 0, 0, 0, 0, 0]

            for batchImage, batchLabel, batchDataIndex in dataLoader:
                batchImage = batchImage.float().cuda()
                batchLabel = batchLabel.long().cuda()

                outputClass = self.Net.forward(batchImage)
                rawLoss = self.LossFunction(outputClass, batchLabel)
                
                loss, lLoss = rawLoss
                loss = loss.mean()
                lLoss = lLoss.mean()

                runningLoss += loss.item()
                labelLoss += lLoss.item()

                for i in range(11):
                    if i not in [1, 2]:
                        BinaryClassificationMetric(outputClass[i], batchLabel[:, i], TP, FP, TN, FN, i)
                    else:
                        MultipleClassificationMetric(outputClass[i], batchLabel[:, i], TP, FP, TN, FN, i)

                for i in range(11):
                    outputClass[i] = outputClass[i].softmax(dim = 1).cpu()
                    answer.Outputs[i] = torch.cat((answer.Outputs[i], outputClass[i]), dim = 0)
                answer.Labels = torch.cat((answer.Labels, batchLabel.float().cpu()), dim = 0)
                answer.DataIndexes += batchDataIndex

            for i in range(11):
                if i not in [1, 2]:
                    answer.Accuracy[i], answer.Recall[i], answer.Precision[i], answer.Specificity[i] = ClassificationMetrics(TP[i], FP[i], TN[i], FN[i], self.Epsilon)
                else:
                    a, r, p, s = [0] * len(TP[i]), [0] * len(TP[i]), [0] * len(TP[i]), [0] * len(TP[i])
                    for c in range(len(TP[i])):
                        a[c], r[c], p[c], s[c] = ClassificationMetrics(TP[i][c], FP[i][c], TN[i][c], FN[i][c], self.Epsilon)
                    answer.Accuracy[i], answer.Recall[i], answer.Precision[i], answer.Specificity[i] = np.mean(a), np.mean(r), np.mean(p), np.mean(s)

            loss = (np.array([runningLoss, labelLoss]) * self.BatchSize) / numOfInstance
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

    EvaluateMachineLearningModel(MultiTaskModel,\
                                 args.SaveFolderPath, (args.TrainFolderPath, args.TrainInfoPath), (args.TestFolderPath, args.TestInfoPath),\
                                 name = args.Name)
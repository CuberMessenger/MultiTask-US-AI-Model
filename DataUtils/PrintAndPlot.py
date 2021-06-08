import os
import csv
import torch
import numpy as np
import matplotlib.pylab as plot

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from StatisticsUtils import CalculateAUC, ClassificationMetrics, BinaryClassificationMetric, MultipleClassificationMetric

targetResolutionPerSubFigure = 1080
targetDPI = 200

class SingleTaskClassificationAnswer():
    def __init__(self):
        self.Outputs = torch.Tensor()
        self.Labels = torch.Tensor()
        self.DataIndexes = []
        self.Accuracy = 0
        self.Recall = 0
        self.Precision = 0
        self.Specificity = 0
        self.TrainLosses = None
        self.ValidationLosses = None

class MultiTaskClassificationAnswer():
    def __init__(self):
        self.Outputs = [torch.Tensor()] * 11
        self.Labels = torch.Tensor()
        self.DataIndexes = []
        self.Accuracy = [0] * 11
        self.Recall = [0] * 11
        self.Precision = [0] * 11
        self.Specificity = [0] * 11
        self.TrainLosses = None
        self.TrainLabelLosses = None
        self.ValidationLosses = None
        self.ValidationLabelLosses = None

def DrawPlots(validationFPRs, validationTPRs, validationAUCs,\
              testFPRs, testTPRs, testAUCs,\
              ensembleFPR, ensembleTPR, ensembleAUC,\
              validationAnswers, saveFolderPath, numOfFold):
    gridSize = 2
    targetFigureSize = (targetResolutionPerSubFigure * gridSize / targetDPI, targetResolutionPerSubFigure * gridSize / targetDPI)
    plot.figure(figsize = targetFigureSize, dpi = targetDPI)
    plot.subplot(gridSize, gridSize, 1)
    for i in range(5):
        plot.title("Validation AUC by folds")
        plot.plot(validationFPRs[i], validationTPRs[i], alpha = 0.7, label = ("Fold %d Val AUC = %0.3f" % (i, validationAUCs[i])))
        plot.legend(loc = "lower right")
        plot.plot([0, 1], [0, 1],"r--")
        plot.xlim([0, 1])
        plot.ylim([0, 1.05])
        plot.ylabel("True Positive Rate")
        plot.xlabel("False Positive Rate")

    plot.subplot(gridSize, gridSize, 2)
    for i in range(5):
        plot.title("Test AUC by folds")
        plot.plot(testFPRs[i], testTPRs[i], alpha = 0.7, label = ("Fold %d Test AUC = %0.3f" % (i, testAUCs[i])))
        plot.legend(loc = "lower right")
        plot.plot([0, 1], [0, 1],"r--")
        plot.xlim([0, 1])
        plot.ylim([0, 1.05])
        plot.ylabel("True Positive Rate")
        plot.xlabel("False Positive Rate")

    plot.subplot(gridSize, gridSize, 3)
    plot.title("Test AUC by ensemble")
    plot.plot(ensembleFPR, ensembleTPR, alpha = 0.7, label = "Test AUC = %0.3f" % ensembleAUC)
    plot.legend(loc = "lower right")
    plot.plot([0, 1], [0, 1],"r--")
    plot.xlim([0, 1])
    plot.ylim([0, 1.05])
    plot.ylabel("True Positive Rate")
    plot.xlabel("False Positive Rate")
    plot.savefig(os.path.join(saveFolderPath, "ROCCurvePlot.png"))

    if validationAnswers[0].TrainLosses is None:
        return
    hasLabelLoss = hasattr(validationAnswers[0], "TrainLabelLosses")
    gridSize = 4 if hasLabelLoss else 3
    targetFigureSize = (targetResolutionPerSubFigure * gridSize / targetDPI, targetResolutionPerSubFigure * gridSize / targetDPI)
    plot.figure(figsize = targetFigureSize, dpi = targetDPI)
    for i in range(numOfFold):
        plot.subplot(gridSize, gridSize, i + 1)
        plot.title("Fold %d Losses" % i)
        plot.plot(np.array(validationAnswers[i].TrainLosses), label = "Train Loss")
        plot.plot(np.array(validationAnswers[i].ValidationLosses), label = "Validation Loss")
        plot.legend(loc = "upper right")
        plot.xlabel("Epoch")
        plot.ylabel("Loss")

        if hasLabelLoss:
            plot.subplot(gridSize, gridSize, i + 6)
            plot.title("Fold %d Label Losses" % i)
            plot.plot(np.array(validationAnswers[i].TrainLabelLosses), label = "Train Label Loss")
            plot.plot(np.array(validationAnswers[i].ValidationLabelLosses), label = "Validation Label Loss")
            plot.legend(loc = "upper right")
            plot.xlabel("Epoch")
            plot.ylabel("Loss")
    plot.savefig(os.path.join(saveFolderPath, "LossesPlot.png"))

def SingleTaskEnsembleTest(testAnswers, saveFolderPath):
    foldPredict = np.array([testAnswer.Outputs[:, 1].numpy() for testAnswer in testAnswers])
    label = testAnswers[0].Labels.numpy()
    rawResults = np.mean(foldPredict, axis = 0)
    predict = (rawResults > 0.5).astype(np.int)
    P = (predict == 1).astype(np.int)
    N = (predict == 0).astype(np.int)
    TP = np.sum(P * label)
    FP = np.sum(P * (1 - label))
    TN = np.sum(N * (1 - label))
    FN = np.sum(N * label)
    accuracy, recall, precision, specificity = ClassificationMetrics(TP, FP, TN, FN)
    ensembleAUC, ensembleFPR, ensembleTPR = CalculateAUC(rawResults, label)
    print("\nEnsemble Test Results:")
    print("AUC,%f\nAccuracy,%f\nRecall,%f\nPrecision,%f\nSpecificity,%f" %\
         (ensembleAUC, accuracy, recall, precision, specificity))

    with open(os.path.join(saveFolderPath, "TestResults.csv"), mode = "w", newline = "") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["DataIndex", "Ensembled"])
        for i, dataIndex in enumerate(testAnswers[0].DataIndexes):
            csvWriter.writerow([dataIndex, str(rawResults[i])])
        
    return ensembleAUC, ensembleFPR, ensembleTPR

def MultiTaskEnsembleTest(testAnswers, saveFolderPath):
    foldPredict = np.array([testAnswer.Outputs[0][:, 1].numpy() for testAnswer in testAnswers])
    label = testAnswers[0].Labels[:, 0].numpy()
    rawResults = np.mean(foldPredict, axis = 0)
    predict = (rawResults > 0.5).astype(np.int)
    P = (predict == 1).astype(np.int)
    N = (predict == 0).astype(np.int)
    TP = np.sum(P * label)
    FP = np.sum(P * (1 - label))
    TN = np.sum(N * (1 - label))
    FN = np.sum(N * label)
    accuracy, recall, precision, specificity = ClassificationMetrics(TP, FP, TN, FN)
    ensembleAUC, ensembleFPR, ensembleTPR = CalculateAUC(rawResults, label)
    print("\nEnsemble Test Results:")
    print("AUC,%f\nAccuracy,%f\nRecall,%f\nPrecision,%f\nSpecificity,%f" %\
         (ensembleAUC, accuracy, recall, precision, specificity))
    
    foldPredicts = []
    labels = []
    for i in range(11):
        foldPredict = np.array([testAnswer.Outputs[i].numpy() for testAnswer in testAnswers])
        foldPredict = np.mean(foldPredict, axis = 0)
        label = testAnswers[0].Labels[:, i].numpy()
        foldPredicts.append(foldPredict)
        labels.append(label)

    with open(os.path.join(saveFolderPath, "TestResults.csv"), mode = "w", newline = "") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["DataIndex", \
                            "Malignancy", "", \
                            "Composition", "", "", "", \
                            "Echogenicity", "", "", "", "", \
                            "Shape", "", \
                            "Margin", "", \
                            "IrregularOrIobulated", "", \
                            "ExtraThyroidalExtension", "", \
                            "LargeCometTail", "", \
                            "Macrocalcification", "", \
                            "Peripheral", "", \
                            "Punctate", ""])
        for r, dataIndex in enumerate(testAnswers[0].DataIndexes):
            row = [str(dataIndex)]
            for i in range(11):
                row += list(foldPredicts[i][r, :])
            csvWriter.writerow(row)

    return ensembleAUC, ensembleFPR, ensembleTPR

def SingleTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath):
    numOfFold = len(validationAnswers)
    #Accuracy, Recall, Precision, Specificity, AUC
    validationAverages = [0] * 5
    testAverages = [0] * 5

    validationAUCs = []
    validationFPRs = []
    validationTPRs = []
    testAUCs = []
    testFPRs = []
    testTPRs = []

    print(",,,Validation,,,,,,Test,,,,")
    print("Fold,Accuracy,Recall,Precision,Specificity,AUC,,Accuracy,Recall,Precision,Specificity,AUC,")
    for i in range(numOfFold):
        #Validation
        validationAUC, validationFPR, validationTPR, validationBestThreshold =\
            CalculateAUC(validationAnswers[i].Outputs[:, 1].numpy(), validationAnswers[i].Labels.numpy(), needThreshold = True)
        validationAUCs.append(validationAUC)
        validationFPRs.append(validationFPR)
        validationTPRs.append(validationTPR)

        validationAverages[0] += validationAnswers[i].Accuracy
        validationAverages[1] += validationAnswers[i].Recall
        validationAverages[2] += validationAnswers[i].Precision
        validationAverages[3] += validationAnswers[i].Specificity
        validationAverages[4] += validationAUC

        print("%d," % i, end = "")
        print("%f," % validationAnswers[i].Accuracy, end = "")
        print("%f," % validationAnswers[i].Recall, end = "")
        print("%f," % validationAnswers[i].Precision, end = "")
        print("%f," % validationAnswers[i].Specificity, end = "")
        print("%f,," % validationAUC, end = "")

        #Test
        testAUC, testFPR, testTPR, testBestThreshold =\
            CalculateAUC(testAnswers[i].Outputs[:, 1].numpy(), testAnswers[i].Labels.numpy(), needThreshold = True)
        testAUCs.append(testAUC)
        testFPRs.append(testFPR)
        testTPRs.append(testTPR)

        testAverages[0] += testAnswers[i].Accuracy
        testAverages[1] += testAnswers[i].Recall
        testAverages[2] += testAnswers[i].Precision
        testAverages[3] += testAnswers[i].Specificity
        testAverages[4] += testAUC

        print("%f," % testAnswers[i].Accuracy, end = "")
        print("%f," % testAnswers[i].Recall, end = "")
        print("%f," % testAnswers[i].Precision, end = "")
        print("%f," % testAnswers[i].Specificity, end = "")
        print("%f," % testAUC)

    validationAverages = np.array(validationAverages) / numOfFold
    testAverages = np.array(testAverages) / numOfFold
    print("Average,", end = "")
    for v in validationAverages:
        print("%f," % v, end = "")
    print(",", end = "")
    for v in testAverages:
        print("%f," % v, end = "")
    print()

    ensembleAUC, ensembleFPR, ensembleTPR = SingleTaskEnsembleTest(testAnswers, saveFolderPath)

    DrawPlots(validationFPRs, validationTPRs, validationAUCs,\
              testFPRs, testTPRs, testAUCs,\
              ensembleFPR, ensembleTPR, ensembleAUC,\
              validationAnswers, saveFolderPath, numOfFold)

def MultiTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath):
    numOfFold = len(validationAnswers)
    #Accuracy, Recall, Precision, Specificity, AUC
    validationAverages = [0] * 5
    testAverages = [0] * 5

    validationAUCs = []
    validationFPRs = []
    validationTPRs = []
    testAUCs = []
    testFPRs = []
    testTPRs = []

    print(",,,Validation,,,,,,Test,,,,")
    print("Fold,Accuracy,Recall,Precision,Specificity,AUC,,Accuracy,Recall,Precision,Specificity,AUC,")
    for i in range(numOfFold):
        #Validation
        validationAUC, validationFPR, validationTPR, validationBestThreshold =\
            CalculateAUC(validationAnswers[i].Outputs[0][:, 1].numpy(), validationAnswers[i].Labels[:, 0].numpy(), needThreshold = True)
        validationAUCs.append(validationAUC)
        validationFPRs.append(validationFPR)
        validationTPRs.append(validationTPR)

        validationAverages[0] += validationAnswers[i].Accuracy[0]
        validationAverages[1] += validationAnswers[i].Recall[0]
        validationAverages[2] += validationAnswers[i].Precision[0]
        validationAverages[3] += validationAnswers[i].Specificity[0]
        validationAverages[4] += validationAUC

        print("%d," % i, end = "")
        print("%f," % validationAnswers[i].Accuracy[0], end = "")
        print("%f," % validationAnswers[i].Recall[0], end = "")
        print("%f," % validationAnswers[i].Precision[0], end = "")
        print("%f," % validationAnswers[i].Specificity[0], end = "")
        print("%f,," % validationAUC, end = "")

        #Test
        testAUC, testFPR, testTPR, testBestThreshold =\
            CalculateAUC(testAnswers[i].Outputs[0][:, 1].numpy(), testAnswers[i].Labels[:, 0].numpy(), needThreshold = True)
        testAUCs.append(testAUC)
        testFPRs.append(testFPR)
        testTPRs.append(testTPR)

        testAverages[0] += testAnswers[i].Accuracy[0]
        testAverages[1] += testAnswers[i].Recall[0]
        testAverages[2] += testAnswers[i].Precision[0]
        testAverages[3] += testAnswers[i].Specificity[0]
        testAverages[4] += testAUC

        print("%f," % testAnswers[i].Accuracy[0], end = "")
        print("%f," % testAnswers[i].Recall[0], end = "")
        print("%f," % testAnswers[i].Precision[0], end = "")
        print("%f," % testAnswers[i].Specificity[0], end = "")
        print("%f," % testAUC)

    validationAverages = np.array(validationAverages) / numOfFold
    testAverages = np.array(testAverages) / numOfFold
    print("Average,", end = "")
    for v in validationAverages:
        print("%f," % v, end = "")
    print(",", end = "")
    for v in testAverages:
        print("%f," % v, end = "")
    print()

    ensembleAUC, ensembleFPR, ensembleTPR = MultiTaskEnsembleTest(testAnswers, saveFolderPath)

    DrawPlots(validationFPRs, validationTPRs, validationAUCs,\
              testFPRs, testTPRs, testAUCs,\
              ensembleFPR, ensembleTPR, ensembleAUC,\
              validationAnswers, saveFolderPath, numOfFold)

def ClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath):
    if type(validationAnswers[0]) is SingleTaskClassificationAnswer:
        SingleTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath)
    if type(validationAnswers[0]) is MultiTaskClassificationAnswer:
        MultiTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath)
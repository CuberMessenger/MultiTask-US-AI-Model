import csv
import random
import numpy as np
random.seed(2223)

class Folds:
    def __init__(self, benignPatients, malignantPatients, numOfFold, dataPath):
        self.BenignPatients = benignPatients
        self.MalignantPatients = malignantPatients
        self.NumOfFold = numOfFold
        self.DataPath = dataPath
        self.Ratio = 1 / self.NumOfFold

    def LoadSetData(self, benignStart, benignEnd, malignantStart, malignantEnd, dataPercentage = 1.0):
        setDataIndexList = []
        setLabelList = []

        def LoadBenignOrMalignant(start, end, ps):
            numOfPatients = int((end - start) * dataPercentage)
            for i in range(start, start + numOfPatients):
                setDataIndexList.append(ps[i][0])
                setLabelList.append(ps[i][1:])
        LoadBenignOrMalignant(benignStart, benignEnd, self.BenignPatients)
        LoadBenignOrMalignant(malignantStart, malignantEnd, self.MalignantPatients)

        return np.array(setDataIndexList), np.array(setLabelList)

    def NextFold(self, trainDataPercentage = 1.0):
        fold = {}
        fold["DataPath"] = self.DataPath

        fold["ValidationSetDataIndex"], \
        fold["ValidationSetLabel"] = \
        self.LoadSetData(0, round(len(self.BenignPatients) * self.Ratio), \
                         0, round(len(self.MalignantPatients) * self.Ratio))

        fold["TrainSetDataIndex"], \
        fold["TrainSetLabel"] = \
        self.LoadSetData(round(len(self.BenignPatients) * self.Ratio), len(self.BenignPatients), \
                         round(len(self.MalignantPatients) * self.Ratio), len(self.MalignantPatients), \
                         dataPercentage = trainDataPercentage)

        self.BenignPatients = self.BenignPatients[round(len(self.BenignPatients) * self.Ratio): len(self.BenignPatients)] + \
                              self.BenignPatients[0: round(len(self.BenignPatients) * self.Ratio)]
        self.MalignantPatients = self.MalignantPatients[round(len(self.MalignantPatients) * self.Ratio): len(self.MalignantPatients)] + \
                                 self.MalignantPatients[0: round(len(self.MalignantPatients) * self.Ratio)]
        return fold

    def GetWholeAsTest(self):
        whole = {}
        whole["DataPath"] = self.DataPath
        whole["TestSetDataIndex"], \
        whole["TestSetLabel"] = \
        self.LoadSetData(0, len(self.BenignPatients), \
                         0, len(self.MalignantPatients))
        return whole

def ReadFolds(paths):
    dataPath, infoPath = paths
    patients = []
    with open(infoPath, mode = "r") as f:
        csvFile = csv.reader(f)
        next(csvFile)
        for row in csvFile:
            dataIndex = row[0].split(".")[0]
            label = int(row[1])
            composition = int(row[2])
            echogenicity = int(row[3])
            shape = int(row[4])
            margin = int(row[5])
            irregularOrIobulated = int(row[6])
            extraThyroidalExtension = int(row[7])
            largeCometTail = int(row[8])
            macrocalcification = int(row[9])
            peripheral = int(row[10])
            punctate = int(row[11])
            patients.append((dataIndex,
                             label,
                             composition,
                             echogenicity,
                             shape,
                             margin,
                             irregularOrIobulated,
                             extraThyroidalExtension,
                             largeCometTail,
                             macrocalcification,
                             peripheral,
                             punctate))
    benignPatients = []
    malignantPatients = []
    for patient in patients:
        if patient[1] == 0:
            benignPatients.append(patient)
        else:
            malignantPatients.append(patient)

    random.shuffle(benignPatients)
    random.shuffle(malignantPatients)

    folds = Folds(benignPatients, malignantPatients, 5, dataPath)
    return folds
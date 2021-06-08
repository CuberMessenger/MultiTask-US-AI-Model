import torch.nn as nn

class MultiTaskBlock(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.FC = nn.Linear(inputSize, outputSize)
        self.ReLu = nn.ReLU(inplace = False)
        self.ToLabel = nn.Linear(outputSize, 2)
        self.ToComposition = nn.Linear(outputSize, 4)
        self.ToEchogenicity = nn.Linear(outputSize, 5)
        self.ToShape = nn.Linear(outputSize, 2)
        self.ToMargin = nn.Linear(outputSize, 2)
        self.ToIrregularOrIobulated = nn.Linear(outputSize, 2)
        self.ToExtraThyroidalExtension = nn.Linear(outputSize, 2)
        self.ToLargeCometTail = nn.Linear(outputSize, 2)
        self.ToMacrocalcification = nn.Linear(outputSize, 2)
        self.ToPeripheral = nn.Linear(outputSize, 2)
        self.ToPunctate = nn.Linear(outputSize, 2)

    def forward(self, x):
        x = self.FC(x)
        x = self.ReLu(x)
        label = self.ToLabel(x)
        composition = self.ToComposition(x)
        echogenicity = self.ToEchogenicity(x)
        shape = self.ToShape(x)
        margin = self.ToMargin(x)
        irregularOrIobulated = self.ToIrregularOrIobulated(x)
        extraThyroidalExtension = self.ToExtraThyroidalExtension(x)
        largeCometTail = self.ToLargeCometTail(x)
        macrocalcification = self.ToMacrocalcification(x)
        peripheral = self.ToPeripheral(x)
        punctate = self.ToPunctate(x)
        return [label,
                composition,
                echogenicity,
                shape,
                margin,
                irregularOrIobulated,
                extraThyroidalExtension,
                largeCometTail,
                macrocalcification,
                peripheral,
                punctate]
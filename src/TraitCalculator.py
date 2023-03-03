import numpy as np

from src.Preproccessor import TextPreprocessor
from src.dataHandler import TrainingData
from src.randomForest import TraitModel


class TraitCalculator:
    def __init__(self):
        self.T_Processor = TextPreprocessor()
        self.MlModel = TraitModel()
        self.labels =[]
        self.sentences = []
        self.allTraitLabel=[]


    def loadData(self):
        data = TrainingData()
        data.importToModel()
        sentences = np.array(data.sentences)
        self.labels = [data.openness,data.consciensiosness,data.extraversion,data.agreebleness,data.nurotisicm]
        self.sentences = data.sentences
        self.allTraitLabel = data.allTraitLabel
        return sentences
    def splitDataset(self,features,label):
        X_train, X_test, y_train, y_test = self.MlModel.test_train_split(0.8,features,label)
        return X_train, X_test, y_train, y_test

    def preprocessor(self,features):
        preprocessedText = self.T_Processor.feedPreprocessorAnArray(features)
        paddedArr = self.T_Processor.sequencePadding(preprocessedText)
        return paddedArr

    def TrainModel(self,features,label):
        clf = self.MlModel.randomForest(features,label)
        return clf

    def testClassifier(self,features,label,clf):

        self.MlModel.test_classifiers(features,label,clf)

    def getText(self,para):
        return para.split('.')


    def calculateTraitScores(self, traitdoubleArr):

        traitScores = [0,0,0,0,0]
        summation = 0

        traitArr = traitdoubleArr[0]

        # calculate the total for each trait

        for i in range(len(traitArr)):
            for j in range(len(traitArr[i])) :
                if traitArr[i][j] <0:
                    traitScores[i] = traitScores[i]+ (traitArr[i][j])
                    summation += (1 * traitArr[i][j])
                else:
                    traitScores[i] = traitScores[i] + (traitArr[i][j])
                    summation += traitArr[i][j]
        print(summation)
        return [traitScores[0]/summation,traitScores[1]/summation,traitScores[2]/summation,traitScores[3]/summation,traitScores[4]/summation,]

#       calculate the scores
#         return




# testArr = [[-1, -1,  1,  0, -1],
# [-1 , 1,  0,  0,  1],
# [-1, -1,  0,  1 ,-1],
# [-1 ,-1, 0 , 0, -1],
# [0 ,0 ,1 ,1 ,0]]
#
#
# tr = TraitCalculator()
# print(tr.calculateTraitScores(testArr))






from src.TraitCalculator import TraitCalculator

class ModelMain:
    def __init__(self,sentences):
       self.text = sentences

    def runnerClass(self):
        X_test = [0, 0, 0, 0, 0]
        X_train = [0, 0, 0, 0, 0]

        y_test = [0, 0, 0, 0, 0]
        y_train = [0, 0, 0, 0, 0]

        tc = TraitCalculator()
        tc.loadData()
        print(len(tc.sentences))
        print(len(tc.labels[1]))

        clfs = []

        for i in range(5):
            X_tr, X_te, y_tr, y_te = tc.splitDataset(tc.sentences[i * 2000:((i + 1) * 1900) + 1200],
                                                     tc.labels[i][i * 2000:((i + 1) * 1900) + 1200])
            X_test[i] = X_te
            X_train[i] = X_tr
            y_test[i] = y_te
            y_train[i] = y_tr
            print("reacheed en")

        for i in range(5):
            paddedArr = tc.preprocessor(X_train[i])
            clf = tc.TrainModel(paddedArr, y_train[i])
            clfs.append(clf)
            paddedTest = tc.preprocessor(X_test[i])
            tc.testClassifier(paddedTest, y_test[i], clf)

        predicts = []
        sentence = self.text
        paraArr = tc.getText(sentence)
        paddedUserQuery = tc.preprocessor(paraArr)
        tempArr = []
        for clf in clfs:
            print(clf.predict(paddedUserQuery))
            tempArr.append(clf.predict(paddedUserQuery))
        predicts.append(tempArr)

        finalPredicts = tc.calculateTraitScores(predicts)
        total = 0
        for val in finalPredicts:
            if (val < 0):
                total += (-1 * val)
            else:
                total += val

        print(finalPredicts[0] / total, finalPredicts[1] / total, finalPredicts[2] / total,
              finalPredicts[3] / total,
              finalPredicts[4] / total, )
        return [finalPredicts[0] / total, finalPredicts[1] / total, finalPredicts[2] / total,
              finalPredicts[3] / total,
              finalPredicts[4] / total]


# M = ModelMain(r"I would describe myself as an introverted person who values structure, organization, and attention to detail. While I may not be the most outgoing or social person, I am dedicated to producing high-quality work and ensuring that everything is done to the best of my abilities. I have a strong work ethic and am committed to meeting deadlines and exceeding expectations. I tend to be methodical in my approach and enjoy breaking down complex tasks into smaller, more manageable parts. I am also comfortable working independently and enjoy having the time and space to focus on my work. However, I recognize the importance of collaboration and communication, and I am always willing to work with others to achieve a common goal. Overall, I believe that my introverted nature and conscientiousness make me a reliable and valuable asset to any team.")
#
# M.runnerClass()
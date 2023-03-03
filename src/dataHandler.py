import csv

class TrainingData:
    sentences = []
    openness =[]
    consciensiosness = []
    extraversion = []
    agreebleness =[]
    nurotisicm = []
    allTraitLabel = []


    def exportToCSV(self):

        # Open the .txt file for reading
        with open(r'D:\Level 5 - 2nd Sem\LoopsImplementation\datasets\negative\nNeg.txt','r') as file:
            lines = file.readlines()

        with open('D:\Level 5 - 2nd Sem\LoopsImplementation\datasets\oceanTraitDataset.csv', 'a', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the header row to the .csv file
            # writer.writerow(['sentence', 'O','C','E','A','N'])

            # Write the contents of the list of lines to the .csv file, with a value of 0 in the second column
            for line in lines:
                writer.writerow([line.strip(), 0,0,0,0,-1])

    def importToModel(self):

        with open('oceanTraitDataset.csv') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                self.sentences.append(row[0])
                self.openness.append(int(row[1]))
                self.consciensiosness.append(int(row[2]))
                self.extraversion.append(int(row[3]))
                self.agreebleness.append(int(row[4]))
                self.nurotisicm.append(int(row[5]))

                tempArr = [int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5])]
                self.allTraitLabel.append(tempArr)

    def getText(self, parahgraph):
        splitedText = parahgraph.split('.')
        return splitedText





# data = TrainingData()
# data.exportToCSV()
# # data.importToModel()
# # print(len(data.sentences))
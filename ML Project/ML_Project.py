import string
import random
import csv
import Network as nn

def checkPassword(password):
    numOfChars = 0
    numOfCaps = 0
    numOfLower = 0
    numOfSymbols = 0
    numOfNumbers = 0
    for x in range(len(password)):
        numOfChars += 1
        if password[x].isupper():
            numOfCaps += 1
        if password[x].islower():
            numOfLower += 1
        if not password[x].isalnum():
            numOfSymbols += 1
        if password[x].isdigit():
            numOfNumbers += 1
    return [numOfCaps, numOfLower, numOfSymbols, numOfNumbers, numOfChars]

  
def isValid(conditions, limits):
    #1 cap 1 lower 1 symbol 1 number 12 length
    if conditions[0] >= limits[0] and conditions[1] >= limits[1] and conditions[2] >= limits[2] and conditions[3] >= limits[3] and conditions[4] > limits[4]:
        return 1
    return 0


def generatePasswords(numOfPasswords, fileName, limits):
    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["caps", "lower", "symbols", "numbers", "length", "valid"]
        
        writer.writerow(field)
        for i in range(numOfPasswords):
            res = ''.join(random.choices(string.digits + string.ascii_letters + string.punctuation, k = random.randrange(1, 25)))
            print(res)
            ps = checkPassword(res)
            writer.writerow([ps[0],ps[1],ps[2],ps[3], ps[4], isValid(ps,limits) ])



def userMode(modelName):
    password = input("Enter Test Password ")
    split = checkPassword(password)
    nn.userPassword(split,modelName)

    
limits = [1,1,1,1,12]
userMode('test_model')

limits = [4,1,1,2,12]
#generatePasswords(10000, 'train.csv',limits)
#generatePasswords(1000, 'test.csv',limits)

#nn.trainNetwork('prodModel','train.csv')

#nn.testModel('prodModel', 'train.csv')

#userMode('prodModel')

#nn.useGPU()
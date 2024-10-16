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

  
def isValid(conditions):
    if conditions[0] >= 1 and conditions[1] >= 1 and conditions[2] >= 1 and conditions[3] >= 1 and conditions[4] > 12:
        return 1
    return 0


def generatePasswords(numOfPasswords):
    with open('profiles1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["caps", "lower", "symbols", "numbers", "length", "valid"]
        writer.writerow(field)
        for i in range(numOfPasswords):
            res = ''.join(random.choices(string.digits + string.ascii_letters + string.punctuation, k = random.randrange(1, 25)))
            print(res)
            ps = checkPassword(res)
            writer.writerow([ps[0],ps[1],ps[2],ps[3], ps[4], isValid(ps) ])



#generatePasswords(1000000)

nn.trainNetwork()
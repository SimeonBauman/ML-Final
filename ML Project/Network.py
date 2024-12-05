
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
 
def trainNetwork():
    # load the dataset, split into input (X) and output (y) variables
    dataset = np.loadtxt('tenThous.csv', delimiter=',',skiprows=1)
    X = dataset[:,0:5]
    y = dataset[:,5]
 
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
 
    # define the model
    model = nn.Sequential(
        nn.Linear(5, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1),
        nn.Sigmoid()
    )
    print(model)
 
    # train the model
    loss_fn   = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    n_epochs = 100
    batch_size = 10
 
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')
 
    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accuracy {accuracy}")
    
    torch.save(model,'test_model')
    
def testModel():
    model = torch.load('test_model', weights_only=False)
    
    dataset = np.loadtxt('million.csv', delimiter=',',skiprows=1)
    X = dataset[:,0:5]
    y = dataset[:,5]
 
    X_test = torch.tensor(X, dtype=torch.float32)
    y_test = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # Make predictions with no_grad
    with torch.no_grad():
        y_pred = model(X_test)

    # Compute accuracy
    accuracy = (y_pred.round() == y_test).float().mean()
    print(f"Test Accuracy: {accuracy}")
    
def userPassword(password):
    model = torch.load('test_model', weights_only=False)
    
    dataset = password
    X = password
    
 
    X_test = torch.tensor(X, dtype=torch.float32)
    

    # Make predictions with no_grad
    with torch.no_grad():
        y_pred = model(X_test)
        found = False
        temp = ""
        pred = str(y_pred[0])
        for i in range(len(pred)):
            if pred[i] == ')':
                found = False
                
            if found == True:
                temp += pred[i]
                
            if pred[i] == '(':
                found = True
      
        num = float(temp)       
        if num > .5:
            print("valid")
        else:
            print("invalid")
     
    # Compute accuracy
    
   

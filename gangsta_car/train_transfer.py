import os
import pickle
from sklearn.metrics import  accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from make_lunar_dataset import WaveformDataset, plot_picking_predictions, process_data #IMPORTANT
from models import BiLSTMEventDetector, SignalDetectionV1
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

def load_dataset():
    try:
        with open('lunar_train_test_split.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
            #Create the dataset and dataloader
            train_dataset = WaveformDataset(X_train, y_train)
            test_dataset = WaveformDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader
    except:
        train_loader, test_loader = process_data('sequence')  # Load the dataset if not stored already
        return train_loader, test_loader

def train_transfer(model, lr, epochs, train_loader, device, name='moon_1ch.pth'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  #factor of 0.1 every 10 epochs

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0

        for inputs,targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device) # use gpu

            optimizer.zero_grad()
            targets = targets.unsqueeze(2)  # Shape: [batch_size, sequence_length, 1]

            outputs = model(inputs.unsqueeze(-1))

            #Calculate loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}")
        scheduler.step()
    torch.save(model, os.path.join('models', name))
    return model

def evaluate(model, train_loader, test_loader, criterion):
    model.eval()
    # Initialize metrics
    train_loss = 0.0
    test_loss = 0.0
    train_correct = 0
    test_correct = 0
    train_total = 0
    test_total = 0
    # Evaluate training set
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(2)
            outputs = model(inputs.unsqueeze(-1))
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.numel()
    train_accuracy = train_correct / train_total
    train_loss /= len(train_loader)
    # Evaluate test set
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(2)
            outputs = model(inputs.unsqueeze(-1))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == targets).sum().item()
            test_total += targets.numel()
    test_accuracy = test_correct / test_total
    test_loss /= len(test_loader)
    # Print metrics
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_dataset()

    model = BiLSTMEventDetector(input_channels=1).to(device)

    state_dict = torch.load(os.path.join('models', '1ch.pth'), map_location=device)
    model.load_state_dict(state_dict)

    #Freeze layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    #Train unfrozen layers
    model = train_transfer(
                model,
                lr=0.001,
                epochs=50, 
                train_loader=train_loader,
                device=device,
                )
    #Evaluate
    evaluate(model, train_loader, test_loader, nn.BCELoss())
    plot_picking_predictions(model, test_loader, device)
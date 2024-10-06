import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from make_dataset import WaveformDataset, plot_picking_predictions, process_data
from models import BiLSTMEventDetector, ConvNETEventClassifier
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score


def load_dataset():
    try:
        with open('train_test_split_detection.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
            #Create the dataset and dataloader
            train_dataset = WaveformDataset(X_train, y_train)
            test_dataset = WaveformDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader
    except:
        train_loader, test_loader = process_data('binary', test_percent=0.1)  # Load the dataset if not stored already
        return train_loader, test_loader


def train_detection(channels, lr, epochs, train_loader, test_loader, device, name='signal_detection_1ch.pth'):
    model = ConvNETEventClassifier(input_channels=channels).to(device)  # Load the model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(1)  # Shape: [batch_size, 1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(test_loader)

        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, LR: {optimizer.param_groups[0]['lr']}")

    torch.save(model.state_dict(), os.path.join('models', name))
    return model

def train_picking(channels, lr, epochs, train_loader, test_loader, device, name='bilstm_event_detector.pth'):
    model = BiLSTMEventDetector(input_channels=channels).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Adjust targets shape to match outputs
            targets = targets.unsqueeze(2)  # Shape: [batch_size, sequence_length, 1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(2)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(test_loader)

        # Step the scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, LR: {optimizer.param_groups[0]['lr']}")

    torch.save(model.state_dict(), os.path.join('models', name))
    return model

def evaluate(model, train_loader, test_loader, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    model.eval()

    def evaluate_loader(loader, loader_name=""):
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1) #2 for lstm
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

                # Convert predictions to binary
                predicted = (outputs > 0.5).float()

                all_predictions.append(predicted.view(-1).cpu().numpy())
                all_targets.append(targets.view(-1).cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        # Calculate metrics
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)

        avg_loss = total_loss / len(loader)

        return avg_loss, precision, recall, f1

    # Evaluate train set
    train_loss, train_precision, train_recall, train_f1 = evaluate_loader(train_loader, "Train")

    # Evaluate test set
    test_loss, test_precision, test_recall, test_f1 = evaluate_loader(test_loader, "Test")

    print(f"Train Loss: {train_loss:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_dataset()
    # model = train_detection(channels=1,
    #                         lr=0.001,
    #                         epochs=100,
    #                         train_loader=train_loader,
    #                         test_loader=test_loader,
    #                         device=device,)

    model = ConvNETEventClassifier(input_channels=1).to(device)

    state_dict = torch.load(os.path.join('models', 'signal_detection_1ch.pth'), map_location=device)
    model.load_state_dict(state_dict)

    # model = train_picking(channels=1,
    #               lr=0.001,
    #               epochs=200,
    #               train_loader=train_loader,
    #               test_loader=test_loader,
    #               device=device,
    #               )
    #Evaluate
    # model = BiLSTMEventDetector(input_channels=1).to(device)

    # state_dict = torch.load(os.path.join('models', '1ch_single.pth'), map_location=device)
    # model.load_state_dict(state_dict)

    evaluate(model, train_loader, test_loader, nn.BCELoss(), device)
    plot_picking_predictions(model, test_loader, device, labels_type='binary', num_samples=10)
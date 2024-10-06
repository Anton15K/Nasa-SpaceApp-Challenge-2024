import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from make_dataset import WaveformDataset, plot_picking_predictions, process_data
from models import BiLSTMEventDetector, SignalDetectionV1
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score

def load_dataset():
    try:
        with open('train_test_split.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
            #Create the dataset and dataloader
            train_dataset = WaveformDataset(X_train, y_train)
            test_dataset = WaveformDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader
    except:
        train_loader, test_loader = process_data('single', test_percent=0.1)  # Load the dataset if not stored already
        return train_loader, test_loader


def train_detection(channels, lr, epochs, name='signal_detection.pth'):
    train_loader, test_loader = process_data('binary')  # Load the dataset
    model = SignalDetectionV1(input_channels=channels)  # Load the model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            targets = targets.unsqueeze()  # Shape: [32, 1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    torch.save(model.state_dict(), os.path.join('models', name))
    evaluate(model, train_loader, test_loader, criterion)

def train_picking(channels, lr, epochs, train_loader, device, name='bilstm_event_detector.pth'):
    model = BiLSTMEventDetector(input_channels=channels).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  #factor of 0.1 every 10 epochs

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

        # Step the scheduler
        if epoch > 160:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, LR: {scheduler.get_last_lr()[0]}") #, LR: {scheduler.get_last_lr()[0]}

    torch.save(model.state_dict(), os.path.join('models', name))
    return model

def evaluate(model, train_loader, test_loader, criterion, labels_type='sequence', device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    model.eval()

    def evaluate_loader(loader, loader_name=""):
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(2)
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

        # Convert positive predictions to seconds for reporting
        sample_rate = 600 / 60  # 600 samples in 60 seconds
        positive_predictions_indices = np.where(all_predictions == 1)[0]
        positive_predictions_seconds = positive_predictions_indices / sample_rate

        print(f"{loader_name} Positive Predictions (in seconds): {positive_predictions_seconds}")

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
    # train_detection(3, 0.001, 10)
    # model = train_picking(channels=1,
    #               lr=0.001,
    #               epochs=200,
    #               train_loader=train_loader,
    #               device=device,
    #               )
    #Evaluate
    model = BiLSTMEventDetector(input_channels=1).to(device)

    state_dict = torch.load(os.path.join('models', '1ch_single.pth'), map_location=device)
    model.load_state_dict(state_dict)

    evaluate(model, train_loader, test_loader, nn.BCELoss(), 'sequence', device)
    plot_picking_predictions(model, test_loader, device)
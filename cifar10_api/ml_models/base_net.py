import torch
import torch.nn as nn
from cifar10_api.ml_models.prediction import Prediction
from cifar10_api.ml_models.utils import TrainResult, TestResult

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class BaseNet(nn.Module):
    def __init__(self, labels, device=torch.device('cpu')) -> None:
        super().__init__()
        self._labels = labels
        self._device = device

    def forward(self, img):
        return None

    def to_device(self):
        return self.to(self._device)

    def base_train_model(self, train_loader, valid_loader, epochs, loss_fn, optim, scheduler=None, in_batch=False,
                         filename=None, verbose=True):
        result = TrainResult()
        for label in self._labels:
            result.class_accuracy[label] = 0.
        prev_loss = torch.inf
        train_samples_per_class = [0 for i in range(len(self._labels))]
        valid_correct_per_class = [0 for i in range(len(self._labels))]
        valid_samples_per_class = [0 for i in range(len(self._labels))]
        # Loop through each epoch
        for i in range(epochs):
            self.train()
            curr_loss = 0.
            # Train the model
            for images, labels in train_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                # Reset the gradient
                self.zero_grad()
                # Predict image labels
                output = self.forward(images)
                pred = torch.argmax(output, dim=1)
                m_labels = labels.view_as(pred)
                for j in range(len(m_labels)):
                    train_samples_per_class[m_labels[j]] += 1
                # Compute the loss, add it to the total loss, and backpropagates
                loss = loss_fn(output, labels)
                curr_loss += loss.item()
                loss.backward()
                # Run the optimizer
                optim.step()
                # Run the scheduler if there is one and if it needs to be run inside a batch
                if in_batch and scheduler:
                    scheduler.step()
            # Run the scheduler if there is one and if it needs to be run outside a batch
            if not in_batch and scheduler:
                scheduler.step()
            result.train_loss.append(curr_loss)
            self.eval()
            curr_loss = 0.
            accuracy = 0
            # Validate the model
            with torch.no_grad():
                for images, labels in valid_loader:
                    images = images.to(self._device)
                    labels = labels.to(self._device)
                    # Predict image labels
                    output = self.forward(images)
                    pred = torch.argmax(output, dim=1)
                    m_labels = labels.view_as(pred)
                    accuracy += pred.eq(m_labels).sum()
                    for j in range(len(m_labels)):
                        valid_samples_per_class[m_labels[j]] += 1
                        if pred[j] == m_labels[j]:
                            valid_correct_per_class[m_labels[j]] += 1
                    # Compute the loss, add it to the total loss
                    loss = loss_fn(output, labels)
                    curr_loss += loss.item()
            result.valid_loss.append(curr_loss)
            accuracy = accuracy / len(valid_loader.dataset) * 100
            # Aggregate the accuracy
            result.accuracy += accuracy
            if verbose:
                print()
                print(f"Eporch {i + 1} / {epochs}: training loss = {result.train_loss[-1]: .3f},\
                        validation loss = {result.valid_loss[-1]: .3f},\
                        accuracy = {accuracy: .3f}%")
            # If the current total loss is smaller than the previous total loss, save the updated model to the file
            if filename and curr_loss < prev_loss:
                if verbose:
                    print('Saving model as the validation loss decreases...')
                torch.save(self.state_dict(), filename)
                prev_loss = curr_loss
        # Compute the mean accuracy
        result.accuracy /= epochs
        for k in range(len(self._labels)):
            # Compute the accuracy per class
            result.class_accuracy[self._labels[k]] = round(
                valid_correct_per_class[k] / valid_samples_per_class[k] * 100, 3)
            # Get the distribution per class for the training set
            result.class_dist.train_dist[self._labels[k]] = round(
                (train_samples_per_class[k] / len(train_loader.dataset) * 100 / epochs), 3)
            # Get the distribution per class for the validation set
            result.class_dist.valid_dist[self._labels[k]] = round(
                (valid_samples_per_class[k] / len(valid_loader.dataset) * 100 / epochs), 3)
        return result

    def base_test_model(self, test_loader, loss_fn, verbose=True):
        result = TestResult()
        for label in self._labels:
            result.class_accuracy[label] = 0.
        self.eval()
        curr_loss = 0.
        accuracy = 0
        test_correct_per_class = [0 for i in range(len(self._labels))]
        test_samples_per_class = [0 for i in range(len(self._labels))]
        # Test the model
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                # Predict image labels
                output = self.forward(images)
                pred = torch.argmax(output, dim=1)
                m_labels = labels.view_as(pred)
                accuracy += pred.eq(m_labels).sum()
                for j in range(len(m_labels)):
                    test_samples_per_class[m_labels[j]] += 1
                    if pred[j] == m_labels[j]:
                        test_correct_per_class[m_labels[j]] += 1
                # Compute the loss, add it to the total loss, and backpropagates
                loss = loss_fn(output, labels)
                curr_loss += loss.item()
        result.test_loss.append(curr_loss)
        accuracy = accuracy / len(test_loader.dataset) * 100
        result.accuracy = accuracy
        if verbose:
            print(f'test loss = {result.test_loss: .3f}, accuracy = {result.accuracy: .3f}%')
        for k in range(len(self._labels)):
            # Compute the accuracy per class
            result.class_accuracy[self._labels[k]] = round(
                test_correct_per_class[k] / test_samples_per_class[k] * 100, 3)
            # Get the distribution per class for the test set
            result.class_dist.test_dist[self._labels[k]] = round(
                test_samples_per_class[k] / len(test_loader.dataset) * 100, 3)
        return result

    def load_model(self, filename, inference_mode=True):
        self.load_state_dict(torch.load(filename))
        if inference_mode:
            self.eval()

    def base_predict(self, img):
        m_img = img.to(self._device)
        output = self.forward(m_img)
        softmax = nn.Softmax(dim=1)
        # Get the probability of each class
        prob_tensor = softmax(output) * 100
        prob_dict = dict()
        for i in range(len(prob_tensor[0])):
            # Round the probabilities to 3 decimal digits
            prob_dict[self._labels[i]] = round(prob_tensor[0, i].item(), 3)
        index = torch.argmax(output, dim=1)
        label = self._labels[index]
        return Prediction(label, prob_dict)

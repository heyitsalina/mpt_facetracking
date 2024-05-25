import torch

# Balanced Accuracy
class BalancedAccuracy:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        # Called at the beginning of each epoch
        self.correct_predictions = torch.zeros(self.nClasses)
        self.total = torch.zeros(self.nClasses)

    def update(self, predictions, groundtruth):
        _, model_predictions = torch.max(predictions, 1)

        # Get the True Negatives and True Positives as well as the total number of datapoints to be predicted
        for i in range(self.nClasses):
            self.correct_predictions[i] += (
                model_predictions[groundtruth == i] == groundtruth[groundtruth == i]
            ).sum()
            self.total[i] += (groundtruth == i).sum()

    def getBACC(self):

        # Calculate accuracy for each included class
        accuracies = self.correct_predictions / self.total
        # Take the mean of the accuracies per class as the balanced accuracy of the predictions
        bacc = torch.mean(accuracies)

        return bacc

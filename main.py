import numpy as np
import pandas as pd

from model.model import Model


def create_model(**kwargs):
    model = Model(**kwargs)
    model.add(number_of_neurons=6)
    model.add(number_of_neurons=3)
    model.add(number_of_neurons=1)

    return model


def main():
    # breast cancer dataset https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
    # for simplicity, we have used only 4 features.
    data = pd.read_csv("data/data.csv", error_bad_lines=False)
    labels = data['diagnosis']  # extracting the target or the groundtruth
    data = data[['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean']]  # extracting the features
    labels = np.array(labels)
    data = np.array(data)

    # creating an empty model object
    learning_rate = 0.01  # we set learning rate as 0.01 so as to slowly converge, we can experiment with different rates.
    epochs = 1000  # we arbitrary chose a high nmber of 1000. Could have increased this number till 15k, 20k, 100k and so on.
    model = create_model(input_shape=4, learning_rate=learning_rate)

    # Training on 400 samples out of 569 samples.
    # This way is not the best way to train the model (since we are updating the weights on every sample, so we are performing stochastic gradient descent)
    # More optimised ways would have been batch/mini-batch/randomised-batch gradient descent.
    for j in range(epochs):
        sum_error = 0  # we will see error on the whole dataset and see how it drops on further epochs.
        for i in range(400):
            # idea is simple, we first perform the forward pass, then update the weights during the backprop step.
            model.feedforward(data[i])
            model.backpropagation(labels[i])
            sum_error += (labels[i] - model.output_matrix[-1][0]) ** 2  # we are taking squares to simply account for magnitude.
        print("epoch ", j, "error is ", sum_error)

    # uncommenting these lines, we can test our model

    # for i in range(410,480):
    #     model.feedforward(data[i])
    #     print(model.output_matrix[-1][0], labels[i])


if __name__ == "__main__":
    main()

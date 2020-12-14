import numpy
import pandas as pd


class Data:

    def load_data(self, **kwargs):
        number_of_rows = kwargs.get('number_of_rows')
        if number_of_rows:
            data = pd.read_csv("data/data.csv", error_bad_lines=False, nrows=number_of_rows)
        else:
            data = pd.read_csv("data/data.csv", error_bad_lines=False)
        data = data.sample(frac=1)
        data = self.convert_data_into_lists(data)

        return data


    def obtain_input_shape(self, data):
        shape = numpy.array(data).shape
        sample_size = shape[0]
        features = shape[1]
        print("number of samples are ", sample_size, " and features/columns ", features)
        return sample_size, features

    def convert_data_into_lists(self, data):
        data_list = []
        for index, row in data.iterrows():
            row_list = []
            column_length = len(row)
            for j in range(column_length):
                row_list.append(row[j])
            data_list.append(row_list)

        return data_list




import numpy as np

def ReadX(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        # only one line that includes everything
        file = f.readlines()

    print(f'#instances: {len(file)}') # 7352 for training set, 2947 for test set

    X_all = []
    for instance in file:
        f = filter(None, instance.split(' '))
        instance_filterd = list(f)
        instance_cleaned = [float(attr.strip()) for attr in instance_filterd]
        X_all.append(instance_cleaned)
    X_all = np.array(X_all)
    print('>>> Reading finished! Data are converted to numpy array.')
    print(f'shape of X: {X_all.shape} ==> each instance has {X_all.shape[1]} attributes.')

    return X_all

def ReadY(path):
    print(f'>>> Reading data from: {path} ...')
    with open(path) as f:
        # only one line that includes everything
        file = f.readlines()

        print(f'#instances: {len(file)}')  # 7352 for training set, 2947 for test set

    y_all = [float(label.strip()) for label in file]
    y_all = np.array(y_all)
    print('>>> Reading finished! Data are converted to numpy array.')
    print(f'shape of y: {y_all.shape}')
    return y_all

if __name__ == '__main__':

    # You can change the path of the files
    X_train = ReadX('UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt')
    y_train = ReadY('UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt')

    X_test = ReadX('UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt')
    y_test = ReadY('UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt')

    #### Do whatever you need in the following ...
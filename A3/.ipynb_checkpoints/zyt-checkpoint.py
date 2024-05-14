import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import random

class DT(object):
    def __init__(self):
        self.method = None          # 0 for random_split and 1 for simple_split
        self.data = None
        self.label = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    def simple_split(self):
        self.train_x = np.array(self.data[:300])
        self.train_y = np.array(self.label[:300])
        self.test_x = np.array(self.data[300:])
        self.test_y = np.array(self.label[300:])

    def random_split(self):
        random_list = random.sample([i for i in range(400)],300)
        self.train_x = np.array([self.data[i] for i in range(400) if i in random_list])
        self.train_y = np.array([self.label[i] for i in range(400) if i in random_list])
        self.test_x = np.array([self.data[i] for i in range(400) if i not in random_list])
        self.test_y = np.array([self.label[i] for i in range(400) if i not in random_list])

    def fit(self,data,label,split_method):
        self.data = data
        self.label = label
        self.method = split_method
        if self.method == 0:
            self.simple_split()
        else:
            self.random_split()
        

    def Decision_tree(self):
        #plot the tree with max_depth = 3 and min_samples_split = 5
        tree = DecisionTreeRegressor(max_depth = 3,min_samples_split = 5,random_state = 0)
        tree.fit(self.train_x,self.train_y)
        plot_tree(tree)
        plt.show()

        #different maximum depth,set least_node_size = 5
        test_error = []
        train_error = []
        x_data = [i for i in range(5,51,5)]
        for i in range(5,51,5):
            tree = DecisionTreeRegressor(max_depth = i,min_samples_split = 5,random_state = 0)
            tree.fit(self.train_x,self.train_y)
            MSE_train = np.sum(np.square(tree.predict(self.train_x) - self.train_y))/300
            MSE_test = np.sum(np.square(tree.predict(self.test_x) - self.test_y))/100
            test_error.append(MSE_test)
            train_error.append(MSE_train)
        plt.legend(['train_error','test_error'])
        plt.plot(x_data,train_error,label = 'train_error')
        plt.plot(x_data,test_error,label = 'test_error')
        plt.xlabel('Number of maximum depth')
        plt.ylabel('MSE')
        plt.title('least_node_size = 5')
        plt.legend()
        plt.show()

        test_error = []
        train_error = []
        x_data = [i for i in range(3,21)]
        for i in range(3,21):
            tree = DecisionTreeRegressor(max_depth = i,min_samples_split = 5,random_state = 0)
            tree.fit(self.train_x,self.train_y)
            MSE_train = np.sum(np.square(tree.predict(self.train_x) - self.train_y))/300
            MSE_test = np.sum(np.square(tree.predict(self.test_x) - self.test_y))/100
            test_error.append(MSE_test)
            train_error.append(MSE_train)
        plt.legend(['train_error','test_error'])
        plt.plot(x_data,train_error,label = 'train_error')
        plt.plot(x_data,test_error,label = 'test_error')
        plt.xlabel('Number of maximum depth')
        plt.ylabel('MSE')
        plt.title('least_node_size = 5')
        plt.legend()
        plt.show()
        
        #different number of least node sizes,set max_depth = 15
        test_error = []
        train_error = []
        x_data = [i for i in range(2,100)]
        for i in range(2,100):
            tree = DecisionTreeRegressor(max_depth = 15,min_samples_split = i,random_state = 0)
            tree.fit(self.train_x,self.train_y)
            MSE_train = np.sum(np.square(tree.predict(self.train_x) - self.train_y))/300
            MSE_test = np.sum(np.square(tree.predict(self.test_x) - self.test_y))/100
            test_error.append(MSE_test)
            train_error.append(MSE_train)
        plt.legend(['train_error','test_error'])
        plt.plot(x_data,train_error,label = 'train_error')
        plt.plot(x_data,test_error,label = 'test_error')
        plt.xlabel('Number of least node size')
        plt.ylabel('MSE')
        plt.title('max_depth = 15')
        plt.legend()
        plt.show()



    def Bagging_of_trees(self):
        #different number of trees,set maximum_depth = 15, least_node_size = 5
        test_error = []
        train_error = []
        x_data = [i for i in range(1,100)]
        for i in range(1,100):
            tree = RandomForestRegressor(max_depth = 15,min_samples_split = 5,random_state = 0
                                        ,n_estimators = i,max_features = None,bootstrap = True)
            tree.fit(self.train_x,self.train_y)
            MSE_train = np.sum(np.square(tree.predict(self.train_x) - self.train_y))/300
            MSE_test = np.sum(np.square(tree.predict(self.test_x) - self.test_y))/100
            test_error.append(MSE_test)
            train_error.append(MSE_train)
        plt.legend(['train_error','test_error'])
        plt.plot(x_data,train_error,label = 'train_error')
        plt.plot(x_data,test_error,label = 'test_error')
        plt.xlabel('Number of trees')
        plt.ylabel('MSE')
        plt.title('max_depth = 15, least_node_size = 5')
        plt.legend()
        plt.show()

        #different maximum depth,set num of trees = 100, least_node_size = 5
        test_error = []
        train_error = []
        x_data = [i for i in range(3,21)]
        for i in range(3,21):
            tree = RandomForestRegressor(max_depth = i,min_samples_split = 5,random_state = 0
                                        ,n_estimators = 100,max_features = None,bootstrap = True)
            tree.fit(self.train_x,self.train_y)
            MSE_train = np.sum(np.square(tree.predict(self.train_x) - self.train_y))/300
            MSE_test = np.sum(np.square(tree.predict(self.test_x) - self.test_y))/100
            test_error.append(MSE_test)
            train_error.append(MSE_train)
        plt.legend(['train_error','test_error'])
        plt.plot(x_data,train_error,label = 'train_error')
        plt.plot(x_data,test_error,label = 'test_error')
        plt.xlabel('Number of maximum dapth')
        plt.ylabel('MSE')
        plt.title('number_of_tree = 100, least_node_size = 5')
        plt.legend()
        plt.show()
        
    def Random_forests(self):
        #different number of trees, set m = 1/3,maximum_depth = 15,least_node_size = 5
        test_error = []
        train_error = []
        x_data = [i for i in range(1,100)]
        for i in range(1,100):
            tree = RandomForestRegressor(max_depth = 15,min_samples_split = 5,random_state = 0
                                        ,n_estimators = i,max_features = 1/3,bootstrap = True)
            tree.fit(self.train_x,self.train_y)
            MSE_train = np.sum(np.square(tree.predict(self.train_x) - self.train_y))/300
            MSE_test = np.sum(np.square(tree.predict(self.test_x) - self.test_y))/100
            test_error.append(MSE_test)
            train_error.append(MSE_train)
        plt.legend(['train_error','test_error'])
        plt.plot(x_data,train_error,label = 'train_error')
        plt.plot(x_data,test_error,label = 'test_error')
        plt.xlabel('Number of tress')
        plt.ylabel('MSE')
        plt.title('m = 1/3, least_node_size = 5, max_depth = 15')
        plt.legend()
        plt.show()

        #different number of m, set number of trees = 100,maximum_depth = 15,least_node_size = 5
        test_error = []
        train_error = []
        x_data = [i/100 for i in range(20,100)]
        for i in range(20,100):
            tree = RandomForestRegressor(max_depth = 15,min_samples_split = 5,random_state = 0
                                        ,n_estimators = 100,max_features = i/100,bootstrap = True)
            tree.fit(self.train_x,self.train_y)
            MSE_train = np.sum(np.square(tree.predict(self.train_x) - self.train_y))/300
            MSE_test = np.sum(np.square(tree.predict(self.test_x) - self.test_y))/100
            test_error.append(MSE_test)
            train_error.append(MSE_train)
        plt.legend(['train_error','test_error'])
        plt.plot(x_data,train_error,label = 'train_error')
        plt.plot(x_data,test_error,label = 'test_error')
        plt.xlabel('Number of m')
        plt.ylabel('MSE')
        plt.title('number_of_tree = 100, least_node_size = 5, max_depth = 15')
        plt.legend()
        plt.show()

    def Random_forests_analysis(self):
        train_x_split = []
        train_y_split = []
        index = [i for i in range(0,300)]
        for i in range(10):
            random.shuffle(index)
            train_x_split.append([self.train_x[j] for j in index[:100]])
            train_y_split.append([self.train_y[j] for j in index[:100]])
        
        x_data = [i for i in range(10,101,10)]
        y_data_var = []
        y_data_bias = [] 
        for i in range(10,101,10):
            #bias_square
            avg_performance = np.zeros(100)
            for j in range(10):
                tree = RandomForestRegressor(max_depth = 15,min_samples_split = 5,random_state = 0
                                            ,n_estimators = i,max_features = None,bootstrap = True)
                tree.fit(train_x_split[j],train_y_split[j])
                avg_performance += tree.predict(self.test_x)
            avg_performance /= 10
            bias = np.sum(np.square(avg_performance - self.test_y))/100
            y_data_bias.append(bias)
            #variance
            variance = np.zeros(100)
            for j in range(10):
                tree = RandomForestRegressor(max_depth = 15,min_samples_split = 5,random_state = 0
                                            ,n_estimators = i,max_features = None,bootstrap = True)
                tree.fit(train_x_split[j],train_y_split[j])
                variance += np.square(tree.predict(self.test_x) - avg_performance)
            variance /= 10
            avg_variance = np.sum(variance)/100
            y_data_var.append(avg_variance)
        plt.plot(x_data,y_data_var)
        plt.xlabel('Numbers of trees')
        plt.ylabel('Variance')
        plt.title('m = None, least_node_size = 5, max_depth = 15')
        plt.show()
        plt.plot(x_data,y_data_bias)
        plt.xlabel('Numbers of trees')
        plt.ylabel('Bias^2')
        plt.title('m = None, least_node_size = 5, max_depth = 15')
        plt.show()
                

data_file = pd.read_csv("Carseats.csv")
table = data_file.values.tolist()
label = np.array([table[i][0] for i in range(400)],dtype=np.float64)
raw_data = np.array([table[i][1:] for i in range(400)])
transpose_data = np.transpose(raw_data)
pd_data = pd.DataFrame(raw_data)

#data analysis
def pre_data_analysis():
    plt.hist(label)
    plt.xlabel('Sales')
    plt.ylabel('frequency')
    plt.show()
    for i in range(10):
        if i != 5 and i != 8 and i != 9:
            plt.hist(np.array(transpose_data[i],dtype=np.int32))
        else:
            plt.hist(transpose_data[i])
        
        if i == 0:
            plt.xlabel('CompPrice')
        if i == 1:
            plt.xlabel('Income')
        if i == 2:
            plt.xlabel('Advertising')
        if i == 3:
            plt.xlabel('Population')
        if i == 4:
            plt.xlabel('Price')
        if i == 5:
            plt.xlabel('ShelveLoc')
        if i == 6:
            plt.xlabel('Age')
        if i == 7:
            plt.xlabel('Education')
        if i == 8:
            plt.xlabel('Urban')
        if i == 9:
            plt.xlabel('US')

        plt.ylabel('frequency')
        plt.show()

#preprocessing(use one-hot coding)
processed_data = pd.get_dummies(pd_data,columns=[5])
for i in range(400):
    if processed_data.loc[i,8] == 'No':
        processed_data.loc[i,8] = 0
    else:
        processed_data.loc[i,8] = 1
    if processed_data.loc[i,9] == 'No':
        processed_data.loc[i,9] = 0
    else:
        processed_data.loc[i,9] = 1
processed_data = np.array(processed_data,dtype=np.int32)


#pre_data_analysis()
dt = DT()
dt.fit(processed_data,label,0)
#dt.Decision_tree()
#dt.Bagging_of_trees()
#dt.Random_forests()
dt.Random_forests_analysis()




    




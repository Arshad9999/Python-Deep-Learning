def threshold(x):
    if x>0:
        return 1
    else:
        return 0
def relu(x):
    if x<0:
        return 0
    else:
        return x
class NeuralNetwork_for_twoinputs_oneoutput:
    def __init__(self,data):
        self.data = data
        self.w1=0.1
        self.w2=0.5
        self.w3=1.3
        self.w4=0.7
        self.w5=1.2
        self.w6=1.8
        self.b1=1.5
        self.b2=1.1
        self.train_errors=[]
        self.test_errors=[]
    def split_data(self):
        X=self.data.drop(columns='t',axis=1)
        Y=self.data['t']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.1, random_state=101)
    def train_neuralnetwork(self,numofepochs):
        self.numofepochs=numofepochs
        self.train_errors=[]
        n=0
        errors_in_each_epoch=[]
        for k in range(numofepochs):
            n += 1
            for i,j in zip(list(self.X_train.index),self.Y_train):
                x1 = self.X_train['x1'][i]
                x2 = self.X_train['x2'][i]
                t=j
                h1=x1*self.w1+x2*self.w3+self.b1
                h2=x1*self.w2+x2*self.w4+self.b1
                h1f=relu(h1)
                h2f=relu(h2)
                y=h1f*self.w5+h2f*self.w6+self.b2
                yf=threshold(y)
                error=(t-yf)**2

                der_of_error_w1=-1*2*(t-yf)*self.w5*x1
                w1_new=self.w1-0.5*der_of_error_w1

                der_of_error_w2=-1*2*(t-yf)*self.w6*x1
                w2_new=self.w2-0.5*der_of_error_w2

                der_of_error_w3=-1*2*(t-yf)*self.w5*x2
                w3_new=self.w3-0.5*der_of_error_w3

                der_of_error_w4=-1*2*(t-yf)*self.w6*x2
                w4_new=self.w4-0.5*der_of_error_w4

                der_of_error_w5=-1*2*(t-yf)*h1f
                w5_new=self.w5-0.5*der_of_error_w5

                der_of_error_w6=-1*2*(t-yf)*h2f
                w6_new=self.w5-0.5*der_of_error_w6

                der_of_error_b1=-1*2*(t-yf)*self.w5
                b1_new=self.b1-der_of_error_b1

                der_of_error_b2=-1*2*(t-yf)
                b2_new=self.b2-der_of_error_b2

                self.w1=w1_new
                self.w2=w2_new
                self.w3=w3_new
                self.w4=w4_new
                self.w5=w5_new
                self.w6=w6_new
                self.b1=b1_new
                self.b2=b2_new

                h1=x1*self.w1+x2*self.w3+self.b1
                h2=x1*self.w2+x2*self.w4+self.b1
                h1f=relu(h1)
                h2f=relu(h2)
                y=h1f*self.w5+h2f*self.w6+self.b2
                yf=threshold(y)
                error=(t-yf)**2
                errors_in_each_epoch.append(error)
            print("Average of Errors of all data points in ",n," epoch is ",sum(errors_in_each_epoch)/len(errors_in_each_epoch))
            self.train_errors.append(sum(errors_in_each_epoch)/len(errors_in_each_epoch))
            errors_in_each_epoch=[]
    def test_neuralnetwork(self):
        n=0
        for i,j in zip(list(self.X_test.index),self.Y_test):
            x1 = self.X_test['x1'][i]
            x2 = self.X_test['x2'][i]
            t=j
            h1=x1*self.w1+x2*self.w3+self.b1
            h2=x1*self.w2+x2*self.w4+self.b1
            h1f=relu(h1)
            h2f=relu(h2)
            y=h1f*self.w5+h2f*self.w6+self.b2
            yf=threshold(y)
            error=(t-yf)**2
            n += 1
            print("Error in ",n," row input of test data is ",error)
            self.test_errors.append(error)
        avgoferror=sum(self.test_errors)/len(self.test_errors)
        print("Average of Errors after testing the data is ",avgoferror)
    def predict(self,x1,x2):
        h1=x1*self.w1+x2*self.w3+self.b1
        h2=x1*self.w2+x2*self.w4+self.b1
        h1f=relu(h1)
        h2f=relu(h2)
        y=h1f*self.w5+h2f*self.w6+self.b2
        yf=threshold(y)
        print("The Prediction for following input is: ",yf)
data_xor_gate = pd.DataFrame([[0,0,0],[0,1,0],[1,0,0],[1,1,1]],columns=['x1','x2','t'])
model1 = NeuralNetwork_for_twoinputs_oneoutput(data_xor_gate)
model1.split_data()
model1.X_train
model1.train_neuralnetwork(5)
model1.test_neuralnetwork()
model1.predict(0,1)

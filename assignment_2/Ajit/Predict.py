import torch
import torch.nn as nn
import torch.utils.data as data

import random
from tqdm import tqdm_notebook as tqdm
random.seed(42)

class DataLoader(data.Dataset):
    def __init__(self,max_train_value=1000, max_n=100,training_set_size=1000):
        random_a_b_s=torch.sort(torch.randint(low=1,high=max_train_value+1,size=(training_set_size,2)))[0].double()
        random_n_s=torch.randint(low=3,high=max_n, size=(training_set_size,1)).double()
        self.lis_expected_outputs=[]
        self.lis_inputs=[]
        for index in range(training_set_size):
            temp_input=[0,0,0]

            temp_input[0]=random_a_b_s[index][0]
            temp_input[1]=random_a_b_s[index][1]
            temp_input[2]=random_n_s[index]
            expected_output=[temp_input[0],temp_input[1]]
            for i in range(int(temp_input[2])-2):
                expected_output.append(expected_output[i]+expected_output[i+1])
            input=torch.tensor(temp_input, dtype=torch.float64).unsqueeze(0)

            output=torch.tensor(expected_output, dtype=torch.float64)
            self.lis_expected_outputs.append(output)
            self.lis_inputs.append(input)
    def __len__(self):
        return len(self.lis_inputs)
    def __getitem__(self, idx):
        dic_in_out={'input':self.lis_inputs[idx] , 'output':self.lis_expected_outputs[idx]}
        return dic_in_out



class MODEL(nn.Module):
    def __init__(self , input_size ,hidden_size,  output_size):
        super(MODEL, self).__init__()
        self.hidden_size=hidden_size
        self.i2h=nn.Linear(input_size+hidden_size, hidden_size,bias=False).double()
        self.i2o=nn.Linear(input_size+hidden_size, output_size,bias=False).double()


    def step(self, input , hidden):
        combined=torch.cat((input,hidden),1)
        hidden=self.i2h(combined)
        output=self.i2o(combined)
        return output,hidden
    def forward(self , input):
        hidden=self.initHidden()
        n=int(input[0][2])
        a=torch.tensor([int(input[0][0])],dtype=torch.float64).unsqueeze(0)
        b=torch.tensor([int(input[0][1])] ,dtype=torch.float64).unsqueeze(0)

        predictions=torch.zeros(size=(1,n)).double()
        predictions[0,0]=a
        predictions[0,1]=b
        _,hidden=self.step(torch.tensor([a[0,0]], dtype=torch.float64).clone().unsqueeze(0),hidden)
        for i in range(n-2):
            temp_output,hidden=self.step(torch.tensor([predictions[0,i+1]] , dtype=torch.float64).clone().unsqueeze(0) , hidden)
            predictions[0,i+2]=temp_output[0]
        return predictions
    def initHidden(self):
        return torch.zeros(1, self.hidden_size).double()


model = torch.load("./data.pt")

a=int(input("Enter a:\n"))
b=int(input("Enter b:\n"))
n=int(input("Enter n:\n"))
output=(model(torch.tensor([a,b,n], dtype=torch.float64).unsqueeze(0)))
print(output.tolist())


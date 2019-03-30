import numpy as np
import torch
from LRM import LinearRegressionModel
model = torch.load('mymodel1.pt')
model.eval()
print("Enter the First Term  A :")
a=input()
print("Enter the First Term  B :")
b=input()
print("Enter the First Term  C :")
n=input()
ans = []
num1=float(a)
num2=float(b)
n=int(n)
for i in range(n):
    x1 = model(torch.tensor([num1,num2],dtype=torch.float))
    x1 = torch.round(x1)
    num1=num2
    num2=x1.data.item()
    ans.append(num2)
print("The next N entries are: \n",ans)

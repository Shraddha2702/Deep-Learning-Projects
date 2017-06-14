#AutoEncoders
#Install Virtual Machine
#Install Ubuntu, Python, Anaconda and other necessary Libraries

#Recommender Systems
#Movie List dataset (MovieLens)

#First we need to preprocess the data, which is common for Boltzmann Machine and AutoEncoders

#Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.util.data
from torch.autograd import Variable

#Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header = None, engine = 'python', encoding = 'latin-1')

users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimeter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimeter = '\t')
test_set = np.array(test_set, dtype = 'int')

#Getting the total number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))

nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))


#Converting the data into an array with users in lines and movies in columns
#Creating a function for that
def convert(data):
   #Rows - Users n Columns - Movies
   #Creating lists of Lists
   new_data = []
   for id_users in range(1,nb_users+1):
      id_movies = data[:,1][data[:,0] == id_users]
      id_ratings = data[:,2][data[:,0] == id_users]

      #Users --->
      #Movies
      #|  Ratings in cells ie intersection of
      #|  rows and columns
      #|

      ratings = np.zeros(nb_movies)
      ratings[id.movies - 1] = id_ratings
      new_data.append(list(ratings))
   return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#------------------------------------------------------------------------------------

#Specific to AutoEncoders

class SAE(nn.Module):
   def __init__(self, ):
      super(SAE, self).__init__()
      self.fc1 = nn.Linear(nb_movies, 20)
      self.fc2 = nn.Linear(20,10)
      self.fc3 = nn.Linear(10,20)
      self.fc4 = nn.Linear(20, nb_movies)
      self.activation = nn.Sigmoid()
   
   def forward(self, x):
      x = self.activation(self.fc1(x))
      x = self.activation(self.fc2(x))
      x = self.activation(self.fc3(x))
      x = self.fc4(x)
      return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters, lr = 0.01, weight_decay = 0.5)

#Training the SAE
for epoch in range(1, nb_users):
   train_loss = 0
   s = 0.
   for id_user in range(nb_users):
      input = Variable(training_set[id_user]).unsqueeze(0)
      target = input.clone()
      if torch.sum(target.data > 0) > 0:
         output = sae(input)
	 loss = criterion(output, target)
 	 mean_corrector = nb_movies/float(torch.sum(target.data > 0)+le-10)
	 loss.backward()
	 train_loss += np.sqrt(loss.data[0] * mean_corrector)
	 s += 1.
 	 optimizer.step()
   print('epoch :'+str(epoch)+' loss :'+str(train_loss/s))

#Testing the SAE
test_loss = 0
s = 0.

for id_user in range(nb_users):
   input = Variable(training_set[id_user]).unsqueeze(0)
   target = Variable(test_set[id_user])
   if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output,target)
      mean_corrector = nb.movies/float(torch.sum(target.data > 0) + le-10)
      test_loss += np.sqrt(loss.data[0] * mean_corrector)
      s += 1.
print('test loss :'str(test_loss/s))
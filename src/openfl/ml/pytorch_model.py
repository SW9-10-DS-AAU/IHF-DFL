import copy
import torch
import random
import numpy as np
from web3 import Web3
import torch.nn as nn
import torch.optim as optim
DEVICE = torch.device("cpu")
from termcolor import colored
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, random_split

# Added
RNG = np.random.default_rng()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bad_c  = "#d62728"
free_c = "#9467bd"
colors.remove(bad_c)
colors.remove(free_c)

class Participant:
    def __init__(self, _id, _train, _val, _model, _optimizer, _criterion,
                 _attitude, _default_collateral, _max_collateral,
                 _attitudeSwitch=1, number_of_participants=None):
        self.id = _id
        self.train = _train
        self.val  = _val
        self.model = _model
        self.previousModel = copy.deepcopy(_model)
        self.modelHash = Web3.solidity_keccak(['string'],[str(_model)]).hex()
        self.optimizer = _optimizer
        self.criterion = _criterion
        self.userToEvaluate = []
        self.currentAcc = 0
        self.attitude = "good"
        self.futureAttitude = _attitude
        self.attitudeSwitch = _attitudeSwitch
        self.hashedModel = None
        self.address = None
        self.privateKey = None
        self.isRegistered = False
        # Old:  self.collateral = _default_collateral + np.random.randint(0,int(_max_collateral-_default_collateral))
        # ---- collateral (handles huge ranges; avoids int32 cap) ----
        lo = int(_default_collateral)
        hi = int(_max_collateral)
        if hi < lo:
            raise ValueError(f"max_collateral ({hi}) must be >= default_collateral ({lo})")

        diff = hi - lo
        jitter = int(RNG.integers(0, np.int64(diff), dtype=np.int64)) if diff > 0 else 0
        self.collateral = lo + jitter

        # ---- secret (big nonce) ----
        self.secret = int(RNG.integers(0, np.int64(10 ** 18), dtype=np.int64))
        # self.secret = np.random.randint(0,int(1e18))

        self.color = get_color(number_of_participants, self.attitude)
        self.roundRep = 0

        self.disqualified = False

        # INTERFACE VARIABLES
        self._accuracy = []
        self._loss = []
        self._globalrep = [self.collateral]
        self._roundrep = []
        
          
class Net_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


        
class PytorchModel:
    def __init__(self, DATASET, _goodParticipants, _totalParticipants, epochs, batchsize, default_collateral, max_collateral):
        self.DATASET = DATASET
        if self.DATASET == "mnist":
            self.global_model = Net_MNIST().to(DEVICE)
        else:
            self.global_model = Net_CIFAR().to(DEVICE)
        
        self.NUMBER_OF_CONTRIBUTERS = _totalParticipants
        self.NUMBER_OF_BAD_CONTRIBUTORS = 0
        self.NUMBER_OF_FREERIDER_CONTRIBUTORS = 0
        self.NUMBER_OF_INACTIVE_CONTRIBUTORS = 0
        self.DATA = None
        
        
                
        self.participants = []
        self.disqualified = []
        self.EPOCHS = epochs
        self.BATCHSIZE = batchsize
        self.train, self.val, self.test = self.load_data(self.NUMBER_OF_CONTRIBUTERS, _print=True)
        self.default_collateral = default_collateral
        self.max_collateral = max_collateral
        loss, accuracy = test(self.global_model,self.test,DEVICE)
        
        # INTERFACE VARIABLES
        self.accuracy = [accuracy]
        self.loss = [loss]
        
        self.round = 1
        print("===================================================================================")
        print("Pytorch Model created:\n")
        print(str(self.global_model))
        print("\n===================================================================================")
        
        for i in range(_goodParticipants):
            if self.DATASET == "mnist":
                _model = Net_MNIST().to(DEVICE)
            else:
                _model = Net_CIFAR().to(DEVICE)
            
            optimizer = optim.SGD(_model.parameters(), lr=0.001, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            _attitude = "good"
                
            self.participants.append(Participant(i, 
                                                 self.train[i], 
                                                 self.val[i], 
                                                 _model, 
                                                 optimizer, 
                                                 criterion,
                                                 _attitude,
                                                 self.default_collateral,
                                                 self.max_collateral,
                                                 None,
                                                 len(self.participants)
                                                ))
            print("Participant added: {} {}".format(gb(_attitude.upper()[0]+_attitude[1:]), gb("User")))
    
            
    def add_participant(self, _attitude, _attitudeSwitch=1):
        
        _train, _val, _test = self.load_data(self.NUMBER_OF_CONTRIBUTERS)
        
        if self.DATASET == "mnist":
            _model = Net_MNIST().to(DEVICE)
        else:
            _model = Net_CIFAR().to(DEVICE)
            
        optimizer = optim.SGD(_model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        if _attitude == "bad":
            self.NUMBER_OF_BAD_CONTRIBUTORS +=1
        if _attitude == "freerider":
            self.NUMBER_OF_FREERIDER_CONTRIBUTORS +=1
        if _attitude == "inactive":
            self.NUMBER_OF_INACTIVE_CONTRIBUTORS +=1
        l = len(self.participants)
        self.participants.append(Participant(len(self.participants), 
                                             _train[l], 
                                             _val[l], 
                                             _model, 
                                             optimizer, 
                                             criterion,
                                             _attitude,
                                             self.default_collateral,
                                             self.max_collateral,
                                             _attitudeSwitch,
                                             len(self.participants)
                                            ))
        
        print("Participant added: {:<9} {}".format(rb(_attitude.upper()[0]+_attitude[1:]), rb("User")))
        
        
        
    def load_data(self, NUM_CLIENTS, _print=False):
        if self.DATA:
            return self.DATA
        
        if self.DATASET == "cifar-10":
            transform = transforms.Compose(
                          [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                        )
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = CIFAR10("./data", train=True, download=True, transform=transform)
            testset = CIFAR10("./data", train=False, download=True, transform=transform_test)
        else:
            trainset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
            testset = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
            
        
        if _print:
            print("Data Loaded:")
            print("Nr. of images for training: {:,.0f}".format(len(trainset)))
            print("Nr. of images for testing:  {:,.0f}\n".format(len(testset)))

        # Split training set into partitions to simulate the individual dataset
        partition_size = len(trainset) // NUM_CLIENTS
        lengths = [partition_size] * NUM_CLIENTS
        
        images_needed = partition_size * NUM_CLIENTS
        if images_needed < len(trainset):
            trainset,_ = random_split(trainset, [images_needed, len(trainset)-images_needed])
        
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []
        for ds in datasets:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
            trainloaders.append(DataLoader(ds_train, batch_size=self.BATCHSIZE, shuffle=True))
            valloaders.append(DataLoader(ds_val, batch_size=self.BATCHSIZE))
        testloader = DataLoader(testset, batch_size=self.BATCHSIZE)
        self.DATA = (trainloaders, valloaders, testloader)
        return trainloaders, valloaders, testloader
    
    
    
    def federated_training(self):
        print(b("\n=========================== FEDERATED LEARNING START =============================="))
        for user in self.participants:
            
            # Special users do not need to train
            if user.attitude == "inactive":
                loss, accuracy = test(user.model, user.val, DEVICE)
                print("{:<17} {} | Epoch -- | Accuracy {:>3.0f} % | Loss {:>6,.2f}".format("Account inactive:  ",
                                                                                 user.address[0:16]+"...",
                                                                                 accuracy*100, loss,))
                continue
            if user.attitude == "freerider":
                loss, accuracy = test(user.model, user.val, DEVICE)
                print("{:<17} {} | Epoch -- | Accuracy {:>3.0f} % | Loss {:>6,.2f}".format("Account freeriding:",
                                                                                 user.address[0:16]+"...",
                                                                                 accuracy*100, loss,))
                continue
            if user.attitude == "bad":
                loss, accuracy = test(user.model, user.val, DEVICE)
                print("{:<17} {} | Epoch -- | Accuracy {:>3.0f} % | Loss {:>6,.2f}".format("Account malicious: ",
                                                                                 user.address[0:16]+"...",
                                                                                 accuracy*100, loss,))
                continue
  
            trainloader = user.train
            valloader   = user.val
            testloader  = self.test

            for epoch in range(1):
                train(user.model, trainloader, self.EPOCHS, DEVICE)
                loss, accuracy = test(user.model, valloader, DEVICE)
                user.currentAcc = accuracy
                _dataload = (epoch+1,accuracy*100,loss)
                print("{:<17} {} | Epoch {:>2} | Accuracy {:>3.0f} % | Loss {:>6,.2f}".format("Account training:  ",
                                                                                user.address[0:16]+"...",
                                                                                *_dataload))
            
            
            loss, accuracy = test(user.model, testloader, DEVICE)
            user._accuracy.append(accuracy)
            user._loss.append(loss)
            
            _dataload = (epoch+1,accuracy*100,loss)
            
            print(green("{:<17} {} | Epoch {:>2} | Accuracy {:>3.0f} % | Loss {:>6,.2f}".format("Account testing:   ",
                                                                                user.address[0:16]+"...",
                                                                                *_dataload)))

            user.hashedModel = self.get_hash(user.model.state_dict()) # OH NO
            print("-----------------------------------------------------------------------------------")

        print(b("=========================== FEDERATED LEARNING END ================================\n"))
        
    
    
    def let_malicious_users_do_their_work(self):
        for i in range(len(self.participants)):
            if self.participants[i].attitude == "bad":                
                print(red("Address {} going to provide random weights".format(self.participants[i].address[0:16]+"...")))
                manipulated_state_dict = manipulate(self.participants[i].model)
                self.participants[i].model.load_state_dict(manipulated_state_dict)
                self.participants[i].hashedModel = self.get_hash(self.participants[i].model.state_dict())
                loss, accuracy = test(self.participants[i].model, self.test, DEVICE)
                print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                self.participants[i].address[0:16]+"...",
                                                                                accuracy*100))
    
    
    
    def update_users_attitude(self):
        for user in self.participants:
            if user.attitudeSwitch == self.round \
                and user.attitude != user.futureAttitude:
                print(rb("Address {} going to switch attitude to {}".format(user.address[0:16]+"...",
                                                                            user.futureAttitude)))
                user.attitude = user.futureAttitude
                user.color = get_color(None, user.attitude)
    
    
    
    def let_freerider_users_do_their_work(self):
        for user in self.participants:
            if user.attitude == "freerider":
              
                # Freerider has no data and must therefore provide something random
                # After first round freerider can copy other participants
                if self.round == 1:
                    print(red("Account {} going to provide ".format(user.address[0:8]+"...") \
                                  + "random weights; starts copycat-ing " \
                                  + "next round"))
                    
                    new_state_dict = manipulate(copy.deepcopy(user.model)) 
                else:
                    foreign_model = copy.deepcopy(self.participants[0].previousModel)
                    new_state_dict = foreign_model.state_dict()
                    
                user.model.load_state_dict(new_state_dict)

                if self.round > 1:
                    print(red("Address {} going to add random noise to weights".format(user.address[0:16]+"...")))
                    user.model.load_state_dict(add_noise(copy.deepcopy(user.model)))
                    
                user.hashedModel = self.get_hash(user.model.state_dict())
                loss, accuracy = test(user.model, self.test, DEVICE)
                print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                user.address[0:16]+"...",
                                                                                accuracy*100))
                
            
    
    
    def the_merge(self, _users):
        ids, client_models = [], []
        for u in _users:
            ids.append(u.id)
            client_models.append(u.model)
            print("Account {} participating in merge".format(u.address[0:16]+"..."))
            #print(test(c[1],self.test,DEVICE))
            
        
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0) #FedAvg
        self.global_model.load_state_dict(global_dict)
        
        loss, accuracy = test(self.global_model,self.test,DEVICE)
        self.accuracy.append(accuracy)
        self.loss.append(loss)
        print("-----------------------------------------------------------------------------------")
        print(b("Merged Model: Accuracy {:>3.0f} % | Loss {:>6,.2f}".format(accuracy*100,loss)))

        for u in self.participants:
            u.previousModel = copy.deepcopy(u.model) #the model from this round
            u.model.load_state_dict(self.global_model.state_dict()) #the global model
           
        print("-----------------------------------------------------------------------------------\n")

    
    
    def exchange_models(self):
        print("Users exchanging models...")
        for user in self.participants:
            user.userToEvaluate = []
            for j in self.participants:
                if user.model == j.model:
                    continue
                if j.model in user.userToEvaluate:
                    continue
                user.userToEvaluate.append(j)
        print("-----------------------------------------------------------------------------------")
    
    
    
    def verify_models(self, on_chain_hashes):
        print("Users verifying models...")
        for _user in self.participants:
            _user.cheater = []
            for user in _user.userToEvaluate:  
                if not self.get_hash(user.model.state_dict()) == on_chain_hashes[user.id]:
                    print(red(f"Account {_user.id}: Account {user.address[0:16]}... could not provide the registered model"))
                    _user.cheater.append(user)
                    
        print("-----------------------------------------------------------------------------------")
                 
    
    
    
    def get_hash(self, _state_dict):
        if type(_state_dict) != dict:
            _state_dict = dict(_state_dict)
        _hash_dict = dict()
        for k,v in _state_dict.items():
            _hash_dict[k] = v.numpy().tobytes()
        return Web3.keccak(text=str(hex(hash(frozenset(sorted(_hash_dict.items(), key=lambda x: x[0]))))))
            
            
    
    def evaluation(self):
        print("Users evaluating models...")
                
        count_dq = len(self.disqualified)
        
        feedback_matrix = np.zeros((1,len(self.participants)+count_dq,len(self.participants)+count_dq))[0]
        accuracy_matrix = np.zeros((1, len(self.participants) + count_dq, len(self.participants) + count_dq))[0]
        
        for feedbackGiver in self.participants:                
            valloader = feedbackGiver.val
            bad_att = feedbackGiver.attitude == "bad"
            free_att = feedbackGiver.attitude == "freerider"
            accuracy_last_round = -1
            
            for ix, user in enumerate(feedbackGiver.userToEvaluate):
                if not bad_att and not free_att:
                    _, accuracy = test(user.model, valloader, DEVICE)
                    accuracy *= 100

                if bad_att:
                    feedback_matrix[feedbackGiver.id][user.id] = -1
                    accuracy_matrix[feedbackGiver.id][user.id] = 0
                    
                elif free_att:
                    feedback_matrix[feedbackGiver.id][user.id] = 0
                    if accuracy_last_round == -1:
                        _, accuracy_last_round = test(self.global_model, valloader, DEVICE)  # TODO: Unitest her
                        accuracy_last_round *= 100
                    accuracy_matrix[feedbackGiver.id][user.id] = accuracy_last_round

                elif user in feedbackGiver.cheater:
                    feedback_matrix[feedbackGiver.id][user.id] = -1
                    accuracy_matrix[feedbackGiver.id][user.id] = accuracy

                
                elif accuracy > feedbackGiver.currentAcc - 0.07: # 7% Worse TODO: Evt tweak
                    feedback_matrix[feedbackGiver.id][user.id] = 1
                    accuracy_matrix[feedbackGiver.id][user.id] = accuracy
                
                elif accuracy > feedbackGiver.currentAcc - 0.14: # 14% Worse TODO: Evt tweak
                    feedback_matrix[feedbackGiver.id][user.id] = 0
                    accuracy_matrix[feedbackGiver.id][user.id] = accuracy
                    
                else : # Even Worse
                    feedback_matrix[feedbackGiver.id][user.id] = -1
                    accuracy_matrix[feedbackGiver.id][user.id] = accuracy


            # RESET
            feedbackGiver.userToEvaluate = []
        
        print("FEEDBACK MATRIX:")
        print(feedback_matrix)
        print("-----------------------------------------------------------------------------------\n")
        print("ACCURACY MATRIX:")
        print(accuracy_matrix)
        print("-----------------------------------------------------------------------------------\n")
        return feedback_matrix, accuracy_matrix

    
# PYTORCH FUNCTIONS
def train(
    net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #print("User {}  |  Epoch {}  |  Batches {}".format(user, epochs, len(trainloader)))

    #print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



def test(
    net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


    
def green(text):
    return colored(text, "green")

def gb(string):
    return colored(string, color="green", attrs=["bold"])

def rb(string):
    return colored(string, color="red", attrs=["bold"])

def b(string):
    return colored(string, color=None, attrs=["bold"])

def red(text):
    return colored(text, "red")

def manipulate(model):
    sd=[]
    for i in [val.cpu().numpy() for _, val in model.state_dict().items()]:
        sd.append(i+random.randint(-100,100)/100)
    
    params_dict = zip(model.state_dict().keys(), sd)
    return OrderedDict({k: torch.tensor(v) for k, v in params_dict})

def add_noise(model):
    sd=[]
    l = len([val.cpu().numpy() for _, val in model.state_dict().items()])
    for ii, i in enumerate([val.cpu().numpy() for _, val in model.state_dict().items()]):
        if ii == l - 5 :
            sd.append(i+random.randint(9,10)/1000000)
        else:
            sd.append(i)
        
    params_dict = zip(model.state_dict().keys(), sd)
    return OrderedDict({k: torch.tensor(v) for k, v in params_dict})

def get_color(i, a):
    if a == "bad":
        return bad_c
    if a == "freerider":
        return free_c
    try:
        return colors[i]
    except:
        return None

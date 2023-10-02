import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

if torch.backends.mps.is_available():
    print('use mps')
    my_device = torch.device("mps")
elif torch.cuda.is_available():
    print('use cuda')
    my_device = torch.device("cuda")
else:
    print('use cpu')
    my_device = torch.device("cpu")

def mkdir(mydir):
	try:
		os.mkdir(mydir)
	except:
		print('failed to make directory', mydir)

odir = 'outdir'
mkdir(odir)
#---
# Load the dataset
#---
dataset = torchvision.datasets.CIFAR10(root='./content/', download=True, train = True, transform=None)
testset = torchvision.datasets.CIFAR10(root='./content/', train=False, transform=None)
label_test_np  = np.array(testset.targets)
x_test_np  = np.array(testset.data)
dataset_tens = torch.tensor(dataset.data, requires_grad=False, device=my_device, dtype=torch.float32)
dataset_labels_tens = torch.tensor(dataset.targets, requires_grad=False, device=my_device, dtype=torch.float32)
testset_tens = torch.tensor(testset.data, requires_grad=False, device=my_device, dtype=torch.float32)
testset_labels_tens = torch.tensor(testset.targets, requires_grad=False, device=my_device, dtype=torch.float32)

train_ds, val_ds = torch.split(dataset_tens, [45000, 5000])
train_ds = train_ds / 255
val_ds = val_ds / 255
test_ds = testset_tens / 255
train_ds_labels, val_ds_labels = torch.split(dataset_labels_tens, [45000, 5000])
y_val = F.one_hot(val_ds_labels.long(), 10).type(torch.float32)
y_train = F.one_hot(train_ds_labels.long(), 10).type(torch.float32)
y_test = F.one_hot(testset_labels_tens.long(), 10).type(torch.float32)

train_ds = train_ds.permute(0,3,1,2)
val_ds = val_ds.permute(0,3,1,2)
test_ds = test_ds.permute(0,3,1,2)

test_ds_np      = test_ds.data


print(train_ds.shape)
print(train_ds_labels.shape)
print(val_ds.shape)
print(val_ds_labels.shape)
print(test_ds.shape)
print(testset_labels_tens.shape)

n_train = train_ds.shape[0]
n_test  = test_ds.shape[0]
n_valid = val_ds.shape[0]
sY = train_ds[1]
sX = train_ds.shape[2]
n_class = 10
chan  = 1

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#-------------------
# Residual Block
#-------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 7, stride = 2, padding = 2),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
        self.layer1a = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 7, stride = 2, padding = 2),
                        nn.BatchNorm2d(out_channels))
        
        self.downsample = downsample
        self.relu = nn.LeakyReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
    #----------------------
    # Create Neural Network
    #----------------------

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        
        
        self.conv1 = nn.Conv2d(3,16, 3, bias=True, padding = 1)
        self.BatchNorm1 = nn.BatchNorm2d(16)
        self.rel1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm2 = nn.BatchNorm2d(16)

        self.rl1 = nn.LeakyReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size= 2, stride = 2)

        self.conv3 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm3 = nn.BatchNorm2d(16)
        self.rel3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm4 = nn.BatchNorm2d(16)

        self.rel2 = nn.LeakyReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size= 2, stride = 2)

        self.conv5 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm5 = nn.BatchNorm2d(16)
        self.rel3 = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(16,16, 3, bias=True, padding = 1)
        self.BatchNorm6 = nn.BatchNorm2d(16)

        self.rel4 = nn.LeakyReLU()
        self.lin1 = nn.Linear(1024, 512)
        self.rel5 = nn.LeakyReLU()
        self.lin2 = nn.Linear(512, 10)



    def forward(self, x):
        #print("FORWARD CHK")
        batch_size = x.shape[0]
        chan       = x.shape[1]
        sY         = x.shape[2]
        sX         = x.shape[3]
        
        #print("x SHAPE:" ,x.shape)
        a = self.conv1(x)
        #print("a SHAPE", a.shape)
        b = self.BatchNorm1(a)
        #print("b SHAPE", b.shape)
        c = self.rel1(b)
        #print("C SHAPE", c.shape)
        d = self.conv2(c)
        #print ("D SHAPE", d.shape)
        e = self.BatchNorm1(d)
        #print("E SHAPE", e.shape)
        e[:,0:3,:,:] = e[:,0:3,:,:] + x

        e = self.rl1(e)
        e = self.avgpool1(e)
        #print("BLOCK1FINISH")
        #loss = torch.sum(Y*torch.log(Yhat + .00000000001))

        ee = self.conv3(e)
        #print("EE SHAPE", ee.shape)
        G = self.BatchNorm3(ee)
        #print("G SHAPE", G.shape)
        H = self.rel2(G)
        #print("H RELU", H.shape)
        I = self.conv4(H)
        #print ("I SHAPE", I.shape)
        J = self.BatchNorm4(I)
        #print("J SHAPE", J.shape)

        J = self.rel2(J)
        J = self.avgpool2(J)

        K = self.conv5(J)
        #print("K SHAPE", K.shape)
        L = self.BatchNorm5(K)
        #print("L SHAPE", L.shape)
        M = self.rel3(L)
        #print("M SHAPE", M.shape)
        N = self.conv6(M)
        #print ("N SHAPE", N.shape)
        O = self.BatchNorm6(N)
        #print("O SHAPE", O.shape)

        P = self.rel4(O)
        # P = torch.reshape(x, (batch_size,chan*sY*sX))
        P = P.view(x.size(0), -1)
        #print("P RESHAPE", P.shape)
        P = self.lin1(P)
        #print("P SHAPE", P.shape)
        P = self.rel5(P)
        #print("P SHAPE", P.shape)
        P = self.lin2(P)
       # print("P SHAPE", P.shape)

        
        z = F.softmax(P,dim=1)
        return z




    
    #--------------------
    # Loop Parameters
    #--------------------

num_classes = 10
num_epochs = 20
batch_size = 100
num_batches = 450
learning_rate = 0.1

model = ResNet().to(my_device)

optim = torch.optim.SGD(model.parameters(), lr=learning_rate)  

loss_train_list = []
acc_train_list = []
loss_valid_list = []
acc_valid_list = []
loss_test_list = []
acc_test_list = []

predicted_testlabel_list = []


# Train the model
for epoch in range(num_epochs):
    correct = 0
    
    print('-----')
    #print('epoch', epoch)
    
    epoch_loss = 0.0
   #TRAIN LOOP ------------------------------------------------------------ 
    for batch in range(num_batches):
        
        #print('epoch', epoch, 'batch', batch)
        
        # reset the optimizer for gradient descent
        optim.zero_grad()
        
        # start / end indices of the data
        sidx =  batch    * batch_size
        eidx = (batch+1) * batch_size
        
        # grab the data and labels for the batch
        X = train_ds[sidx:eidx]
        Y = y_train[sidx:eidx]
        
        # run 
        #print("BATCH SHAPE", X.shape)
        Yhat = model(X)
        #print("Y Shape" ,Y.shape)
        #print("Yhat Shape", Yhat.shape)
        loss = F.cross_entropy(Yhat,Y)

        # gradient descent
        loss.backward()
        optim.step()
    
        # keep track of the loss
        loss_np = loss.detach().cpu().numpy()
        epoch_loss = epoch_loss + loss_np
        _, predicted = torch.max(Yhat.data, 1)
        _, Y = torch.max(Y, 1)

        correct += (predicted == Y).sum().item()
        
    epoch_loss = epoch_loss / n_train
    loss_train_list.append(epoch_loss)
    acc_train = (100* correct / n_train)
    acc_train_list.append(acc_train)
    print(correct, "correct")
    print(num_batches, "num_batches")
    print('epoch %d loss %f' % (epoch+1, epoch_loss))
    print('accuracy for epoch %d : %f' % (epoch+1, acc_train) )
    

    print('training cycle --------------------')
#VALID LOOP ------------------------------------------------------------
    with torch.no_grad():  
        correct = 0
        v_epoch_loss = 0
        r = int(n_valid/batch_size)
        for batch in range(r):
            
            #print('epoch', epoch, 'batch', batch)
            
            # reset the optimizer for gradient descent
            
            # start / end indices of the data
            sidx =  batch    * batch_size
            eidx = (batch+1) * batch_size
            
            # grab the data and labels for the batch
            X = val_ds[sidx:eidx]
            Y = y_val[sidx:eidx]
            
            # run 
            #print("BATCH SHAPE", X.shape)
            Yhat = model(X)
            #print("Y Shape" ,Y.shape)
            #print("Yhat Shape", Yhat.shape)
            loss = F.cross_entropy(Yhat,Y)

            # gradient descent
             
            # keep track of the loss
            v_loss_np = loss.detach().cpu().numpy()
            v_epoch_loss = v_epoch_loss + v_loss_np
            
            _, predicted_val = torch.max(Yhat.data, 1)
            _, Y = torch.max(Y,1)
            correct += (predicted_val == Y).sum().item()

        v_epoch_loss = v_epoch_loss / n_valid
        loss_valid_list.append(v_epoch_loss)
        acc_valid = (100* correct / n_valid)
        acc_valid_list.append(acc_valid)
        print(correct, "correct")
        print(num_batches, "num_batches")
        print('epoch %d loss %f' % (epoch+1, v_epoch_loss))
        print('accuracy for v epoch %d : %f' % (epoch+1, acc_valid) )
       
#TEST LOOP ------------------------------------------------------------

with torch.no_grad():  
    correct = 0
    print("#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#TESTING#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#")
    r = int(n_test/batch_size)
    for batch in range(r):
        
        #print('epoch', epoch, 'batch', batch)
        
        # reset the optimizer for gradient descent
        
        # start / end indices of the data
        sidx =  batch    * batch_size
        eidx = (batch+1) * batch_size
        
        # grab the data and labels for the batch
        X = test_ds[sidx:eidx]
        Y = y_test[sidx:eidx]
        
        # run 
        #print("BATCH SHAPE", X.shape)
        Yhat = model(X)
        #print("Y Shape" ,Y.shape)
        #print("Yhat Shape", Yhat.shape)
        loss = F.cross_entropy(Yhat,Y)

        # gradient descent
            
        # keep track of the loss
        t_loss_np = loss.detach().cpu().numpy()
        t_epoch_loss = epoch_loss + t_loss_np
        
        _, predicted_val = torch.max(Yhat.data, 1)
        _, Y = torch.max(Y,1)
        predicted_testlabel_list.append(predicted_val)
        correct += (predicted_val == Y).sum().item()

    t_epoch_loss = t_epoch_loss / 100
    loss_test_list.append(t_epoch_loss)
    acc_test = (100* correct / n_test)
    acc_test_list.append(acc_test)
    print(correct, "correct")
    print(num_batches, "num_batches")
    print('epoch %d loss %f' % (epoch+1, t_epoch_loss))
    print('accuracy for test epoch %d : %f' % (epoch+1, acc_test) )
    print('TRAIN ACCURACY LIST', len(loss_train_list))
    print('VALID ACCURACY LIST', len(loss_valid_list))
    print('TEST ACCURACY LIST', len(loss_test_list))
    figno = 1
    print('y_test length', len(y_test))
    print('predicted_val length', len(predicted_val))
    matplotlib.use('Agg')
    pred_np = predicted_testlabel_list
    for i in range(50):
        pred = classes[int(predicted_testlabel_list[0][i].item())]
        lab = classes[int(label_test_np[i])]
        acc = acc_test_list[0]
        print('label ',lab) 
        print('predicted ',pred)
        print('accuracy', acc)
        # print('image no',i)
        # print('a',acc_test_list[0])
        # print('p',predicted_val[0])
        # print('y',y_test[0])
        img = x_test_np[i,:,:]
        print('i', i, 'img', img.shape)
        "test1"
        plt.figure(figno)
        "test2"
        plt.title("test %d label %s pred %s accur %.2f" % (i, lab, pred, acc))
        "test3"
        plt.imshow(X=img, cmap='gray', vmin=0, vmax=255)
        "test4"
        #plt.show()
        plt.savefig('%s/%s%05d.png' % (odir,"test_", i))
        "test5"
        plt.close()
        figno+=1

    yplot = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    


    plt.title("ResNet3 Model")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    xx = yplot
    yy = acc_train_list
    yyy = acc_valid_list
    yyyy = np.full(20, acc)
    plt.plot(xx,yy, color = "blue", label = "train")
    plt.plot(xx, yyy, color = "orange", label = "vaid")
    plt.plot(xx, yyyy, color = "purple", label = "test")
    plt.gca().legend(loc = "upper right")
    #plt.plot(loss_valid_list, yplot, color ="blue")
    #plt.axhline(y = acc_test_list[0], color = 'green')
    plt.savefig('%s/%s.png' % (odir, "accuracy"))

    plt.clf()
    plt.close()


    plt.title("ResNet3 Model")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0.015,0.021)
    xx = yplot
    yy = loss_train_list
    yyy = loss_valid_list
    yyyy = np.full(20, loss_test_list[0])
    print(yy)
    print(yyy)
    print(yyyy)
    plt.plot(xx,yy, color = "blue", label = "train")
    plt.plot(xx, yyy, color = "orange", label = "valid")
    plt.plot(xx, yyyy, color = "purple", label = "test")
    #plt.plot(loss_valid_list, yplot, color ="blue")
    #plt.axhline(y = loss_test_list[0], color = 'green')
    plt.gca().legend()
    plt.savefig('%s/%s.png' % (odir, "loss"))
    




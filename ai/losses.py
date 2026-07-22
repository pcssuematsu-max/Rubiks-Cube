"""Local loss functions and probability helpers for Rubiks AI."""

import numpy as np

def sigmoid(x):
    clipped_x = np.clip(x,-60.0,60.0)
    return 1.0 / (1.0 + np.exp(-clipped_x))

class Q_loss:
    def __init__(self,g):
        self.args = np.zeros(0,dtype = 'i')
        self.x = np.zeros(0,dtype = 'f')
        self.rewards = np.zeros(0,dtype = 'f')
        self.maxargs = np.zeros(0,dtype = 'i')
        self.v = np.zeros(0,dtype = 'f')
        self.X = np.zeros(0,dtype = 'i')
        self.g = g

        

    def forward(self,x,args,rewards):
        self.args = args
        self.x = x
        self.rewards = rewards

        self.maxargs = np.argmax(x[:,1:],axis = 0)

        self.X = np.arange(1,x.shape[1])

        self.v = self.x[self.args,self.X - 1] - self.rewards - self.g * self.x[self.maxargs,self.X]


        return np.sum(self.v ** 2)/2
        #return np.sum(huber(self.v))

    def backward(self):
        dO = np.zeros_like(self.x)

        #dv = self.v.copy()
        #dv[dv >= 1] = 1
        #dv[dv <= -1] = -1
        #dO[self.args,self.X - 1] += dv
        
        dO[self.args,self.X - 1] += self.v
        dO[self.maxargs,self.X] -= self.g * self.v
        


        return dO
        
class MSE:
    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.w = np.ones((1,0),dtype = 'f')

    def forward(self,x,t,w = None):
        self.x = x
        self.t = t
        if w is None:
            self.w = np.ones_like(t,dtype = 'f')
        else:
            self.w = w

        return np.sum(self.w * (x - t) ** 2) / 2
    
    def backward(self):
        return self.w * (self.x - self.t)


class BCEWithLogits:
    """Binary cross entropy that takes raw logits and avoids sigmoid saturation in backprop."""

    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.w = np.ones((1,0),dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')

    def forward(self,x,t,w = None):
        self.x = x
        self.t = t
        if w is None:
            self.w = np.ones_like(t,dtype = 'f')
        else:
            self.w = w
        clipped_x = np.clip(x,-60.0,60.0)
        self.y = 1.0 / (1.0 + np.exp(-clipped_x))
        loss = np.maximum(x,0.0) - x * t + np.log1p(np.exp(-np.abs(x)))
        return np.sum(self.w * loss)

    def backward(self):
        return self.w * (self.y - self.t)
	    

class Soft_Target_Cross_Entropy:
    def __init__(self):
        self.y = np.zeros((0,0),dtype = 'f')
        self.t = np.zeros((0,0),dtype = 'f')
        self.w = np.ones((1,0),dtype = 'f')

    def forward(self,x,t,w = None):
        self.t = t
        self.y = softmax(x)
        if w is None:
            self.w = np.ones((1,t.shape[1]),dtype = 'f')
        else:
            self.w = w
        return -np.sum(self.w * t * np.log(self.y + 1.0e-7))

    def backward(self):
        return self.w * (self.y - self.t)


def softmax(x):
    y = x.copy()
    y -= np.max(y,axis = 0)
    y = np.exp(y)
    

    return y / np.sum(y,axis = 0)


def softmax_H(x):
    y = x.copy()
    y -= np.max(y,axis = 1)
    y = np.exp(y)
    

    return y / np.sum(y,axis = 1)    




def cross_entropy(y,t):
    return -np.sum(t * np.where(y != 0 , np.log(y),-100))



        

class Softmax_Cross_Entropy:
    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.Indices = None

    def forward(self,x,args,rewards,Indices,t = None):
        self.Indices = Indices
        self.x = x
        self.y = np.zeros((self.x.shape[0],0),dtype = 'i')
        self.t = np.zeros((self.x.shape[0],0),dtype = 'i')
        for i in range(len(Indices) - 1):
            self.y = np.c_[self.y,softmax(x[:,Indices[i]:Indices[i+1]-1])]

        if t is None:
            self.t = np.zeros_like(self.y,dtype = 'f')
            self.t[args,np.arange(self.t.shape[1])] = 1
        else:
            self.t = t
        return cross_entropy(self.y,self.t)

    def backward(self):
        dO = np.zeros_like(self.x)
        for i in range(len(self.Indices) -1):
            dO[:,self.Indices[i]:self.Indices[i+1] - 1] = self.y[:,self.Indices[i] - i:self.Indices[i+1] - i - 1] - self.t[:,self.Indices[i] - i:self.Indices[i+1] - i - 1]
        return dO

class Myloss:
    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.Indices = None

    def forward(self,x,Indices):

        self.y = np.zeros((1,Indices[-1]),dtype = 'f')
        self.t = np.zeros((1,Indices[-1]),dtype = 'f')
        for i in range(len(Indices) - 1):
            self.y[:,Indices[i]:Indices[i+1]] = softmax_H(x[:,Indices[i]:Indices[i+1]])
            self.t[0,Indices[i+1]-1] = 1.0
        
        return cross_entropy(self.y,self.t)
        
        #self.x = x
        #self.Indices = Indices
        #self.y = np.zeros((1,0),dtype = 'f')
        #for i in range(len(Indices) - 1):
        #    self.y = np.c_[self.y,self.x[:,Indices[i] + 1:Indices[i + 1]] - self.x[:,Indices[i]:Indices[i + 1] - 1]]
        
        #self.y = sigmoid(self.y)

        #return np.sum(self.y)
        

    def backward(self):
        return self.y - self.t
        #dO = np.zeros_like(self.x)
        #for i in range(len(self.Indices) - 1):
        #    dO[:,self.Indices[i]:self.Indices[i + 1] - 1] += self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)] * (1.0 - self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)])
        #    dO[:,self.Indices[i] + 1:self.Indices[i + 1]] -= self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)] * (1.0 - self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)])  
                        
        #return dO


class MyLoss2:
    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.Indices = None


    def forward(self,x,Indices,steps_to_goal = None,gamma = None):
        self.x = x
        self.Indices = Indices
        if gamma is None:
            gamma = 1.0
        if steps_to_goal is None:
            steps_to_goal = self._default_steps_to_goal(Indices)
        dtype = self.x.dtype
        steps_to_goal = np.asarray(steps_to_goal,dtype = dtype).reshape(-1)
        self.y = np.zeros_like(self.x,dtype = dtype)
        self.t = np.zeros_like(self.x,dtype = dtype)
        for i in range(len(Indices) - 1):
            start = Indices[i]
            end = Indices[i + 1]
            if end <= start:
                continue
            self.y[:,start:end] = softmax_H(self.x[:,start:end])
            target = np.asarray(gamma,dtype = dtype) ** steps_to_goal[start:end]
            target_sum = np.sum(target)
            if target_sum <= 0:
                target = np.ones_like(target,dtype = dtype) / max(1,len(target))
            else:
                target = target / target_sum
            self.t[:,start:end] = target.reshape(1,-1)
        return cross_entropy(self.y,self.t)
    

    def backward(self):
        return self.y - self.t

    def _default_steps_to_goal(self, Indices):
        steps = []
        for i in range(len(Indices) - 1):
            start = Indices[i]
            end = Indices[i + 1]
            steps.extend(range(end - start - 1,-1,-1))
        return steps


class MyLoss2Pairwise:
    """Adjacent pairwise margin loss for Search2 value training."""

    def __init__(self, margin = 0.2):
        self.margin = float(margin)
        self.x = np.zeros(0,dtype = 'f')
        self.diffs = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.Indices = None

    def forward(self,x,Indices):
        self.x = x
        self.Indices = Indices
        self.diffs = np.zeros((1,0),dtype = x.dtype)
        for i in range(len(Indices) - 1):
            start = Indices[i]
            end = Indices[i + 1]
            if end - start <= 1:
                continue
            self.diffs = np.c_[self.diffs,self.x[:,start + 1:end] - self.x[:,start:end - 1]]
        self.y = sigmoid(self.diffs - self.margin)
        return np.sum(np.logaddexp(0.0,self.margin - self.diffs))

    def backward(self):
        dO = np.zeros_like(self.x)
        if self.Indices is None:
            return dO
        D = self.y - 1.0
        diff_index = 0
        for i in range(len(self.Indices) - 1):
            start = self.Indices[i]
            end = self.Indices[i + 1]
            count = end - start - 1
            if count <= 0:
                continue
            d = D[:,diff_index:diff_index + count]
            dO[:,start:end - 1] -= d
            dO[:,start + 1:end] += d
            diff_index += count
        return dO

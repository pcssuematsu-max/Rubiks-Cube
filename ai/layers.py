"""Local neural-network layers and activation helpers for Rubiks AI."""

import math

import numpy as np


class Affine:
    def __init__(self,ips,ops):
        self.W = np.random.randn(ops,ips).astype('f') / math.sqrt(ips / 2)
        self.B = np.random.randn(ops).astype('f')
        self.B[:] = 0
        self.dW = np.zeros_like(self.W,dtype = 'f')
        self.dB = np.zeros_like(self.B,dtype = 'f')
        self.x = np.empty((0,0),dtype = 'f')
        self.dx = np.empty((0,0),dtype = 'f')
        self.out = np.empty((0,0),dtype = 'f')


    def forward(self,x):
        self.x = x
        self.out = self.W @ x
        self.out += self.B.reshape(-1,1)

        return self.out

    def backward(self,dO):
        self.dx = self.W.T @ dO
        self.dW = dO @ self.x.T
        self.dB = np.sum(dO,axis = 1)

        return self.dx

def relu(x):
    y = x.copy()
    y[y<0] = 0

    return y

class ReLU:
    def __init__(self):
        self.out = np.empty((0,0),dtype = 'f')

    def forward(self,x):
        self.out = relu(x)
        return self.out

    def backward(self,dO):
        dO = dO * np.sign(self.out)
        return dO

def huber(x):
    y = x ** 2 / 2
    y[y>0.5] = abs(x[y>0.5]) - 0.5

    return y


def hard_sigmoid(x):
    y = x.copy()
    y[y > 1] = 1
    y[y < -1] = -1

    return y

class Hard_Sigmoid:
    def __init__(self):
        self.out = np.empty((0,0),dtype = 'f')

    def forward(self,x):
        self.out = hard_sigmoid(x)
        return self.out

    def backward(self,dO):
        dO[self.out == 1] = 0
        dO[self.out == -1] = 0
        return dO

def sigmoid(x):
    return np.where(x > 40,1.0,np.exp(x) / (1 + np.exp(x)))

class Sigmoid:
    def __init__(self):
        self.out = np.empty((0,0),dtype = 'f')

    def forward(self,x):
        self.out = sigmoid(x)
        return self.out

    def backward(self,dO):
        dO *= self.out * (1.0 - self.out)
        return dO


def stable_sigmoid(x):
    y = np.empty_like(x,dtype = 'f')
    positive = x >= 0
    y[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    y[~positive] = exp_x / (1.0 + exp_x)
    return y


class SiLU:
    """Layer-wise scalar-beta SiLU/Swish: y = x * sigmoid(beta * x)."""

    def __init__(self, beta = 1.0):
        self.beta = np.asarray([beta],dtype = 'f')
        self.dbeta = np.zeros_like(self.beta,dtype = 'f')
        self.x = np.empty((0,0),dtype = 'f')
        self.s = np.empty((0,0),dtype = 'f')
        self.out = np.empty((0,0),dtype = 'f')

    def forward(self,x):
        self.x = x
        self.s = stable_sigmoid(self.beta[0] * x)
        self.out = x * self.s
        return self.out

    def backward(self,dO):
        beta = self.beta[0]
        common = self.s * (1.0 - self.s)
        self.dbeta = np.asarray([np.sum(dO * self.x * self.x * common)],dtype = 'f')
        return dO * (self.s + beta * self.x * common)


class ResidualBlock:
    """Affine -> optional BN -> activation block with shape-safe residual skip."""

    def __init__(self, affine, activation, bn = None):
        self.affine = affine
        self.activation = activation
        self.bn = bn
        self.use_skip = False

    def forward(self, x, loss = False):
        out = self.affine.forward(x)
        if self.bn is not None:
            out = self.bn.forward(out, loss)
        out = self.activation.forward(out)
        self.use_skip = (out.shape == x.shape)
        if self.use_skip:
            return out + x
        return out

    def backward(self, dO):
        skip_grad = dO.copy() if self.use_skip else 0
        d = self.activation.backward(dO)
        if self.bn is not None:
            d = self.bn.backward(d)
        d = self.affine.backward(d)
        return d + skip_grad


class Batch_Normalization:
    def __init__(self,size):
        self.size = size
        self.m = np.zeros(size,dtype = 'f')
        self.s = np.ones(size,dtype = 'f')
        self.g = np.random.uniform(0.5,2,(size,)).astype('f')
        self.b = np.random.uniform(-0.5,0.5,(size,)).astype('f')
        self.dg = None
        self.db = None
        self.m_batch = np.zeros(size,dtype = 'f')
        self.s_batch = np.zeros(size,dtype = 'f')
        self.x_bar = None

        self.N = 0

        self.train_m = None
        self.train_s = None
        self.x = None
        self.y = None

    def forward(self,x,loss = False):
        if loss:
            self.x = x
            self.train_m = np.average(x,axis = 1)
            self.train_s = np.std(x,axis = 1)
            self.x_bar = (x - self.train_m.reshape(-1,1)) / (self.train_s.reshape(-1,1) + 1.0e-7)
            self.y = self.g.reshape(-1,1) * self.x_bar + self.b.reshape(-1,1)

            self.m_batch += self.train_m
            self.s_batch += self.train_s
            self.N += 1

            return self.y

        x_bar = (x - self.m.reshape(-1,1)) / (self.s.reshape(-1,1) + 1.0e-7)
        return self.g.reshape(-1,1) * x_bar + self.b.reshape(-1,1)

    def backward(self,dO):
        self.dg = np.sum(self.x_bar * dO,axis = 1)
        self.db = np.sum(dO,axis = 1)

        return self.g.reshape(-1,1) / (self.train_s.reshape(-1,1) + 1.0e-7) * (dO- (self.db.reshape(-1,1) + self.x_bar * self.dg.reshape(-1,1)) / self.x.shape[1])

    def set_ms(self):
        self.m[:] = self.m_batch / self.N
        self.s[:] = self.s_batch / self.N

        self.N = 0
        self.m_batch[:] = 0
        self.s_batch[:] = 0

def softmax(M):
    X = M.copy()
    X -= np.max(M,axis = -1,keepdims = True)
    X = np.exp(X)

    return X / np.sum(X,axis = -1,keepdims = True)


def softmax_backward(dO,y):
    """Row-wise softmax backward for the last axis."""
    return y * (dO - np.sum(dO * y,axis = -1,keepdims = True))


class Transformer_SelfAttention:
    def __init__(self,ips,mps,ops = None):
        if ops is None:
            ops = mps
            mps = ops
        else:
            mps = mps
        self.ips = int(ips)
        self.mps = int(mps)
        self.ops = int(ops)
        self.WQ = np.random.randn(self.mps,self.ips).astype('f') / math.sqrt(max(self.ips / 2,1))
        self.WK = np.random.randn(self.mps,self.ips).astype('f') / math.sqrt(max(self.ips / 2,1))
        self.WV = np.random.randn(self.ops,self.ips).astype('f') / math.sqrt(max(self.ips / 2,1))
        self.scale = 1.0 / math.sqrt(max(self.mps,1))
        self.x = None
        self.dWQ = np.zeros_like(self.WQ)
        self.dWK = np.zeros_like(self.WK)
        self.dWV = np.zeros_like(self.WV)
        self.Q = None
        self.K = None
        self.V = None
        self.M = None
        self.attended = None
        self.out = None
    
    def forward(self,x):
        self.x = x
        self.Q = self.WQ @ x
        self.K = self.WK @ x
        self.V = x

        score = self.scale * np.einsum('ib,jb->bij',self.Q,self.K)
        self.M = softmax(score)
        self.attended = np.einsum('bij,jb->ib',self.M,self.V).astype('f')
        self.out = (self.WV @ self.attended).astype('f')
        return self.out
    
    def backward(self,dO):
        self.dWV = dO @ self.attended.T
        dA = self.WV.T @ dO
        dM = np.einsum('ib,jb->bij',dA,self.V)
        dS = softmax_backward(dM,self.M)
        dV = np.einsum('bij,ib->jb',self.M,dA)

        dQ = self.scale * np.einsum('bij,jb->ib',dS,self.K)
        dK = self.scale * np.einsum('bij,ib->jb',dS,self.Q)

        self.dWQ = dQ @ self.x.T
        self.dWK = dK @ self.x.T

        return self.WQ.T @ dQ + self.WK.T @ dK + dV


class PieceTokenSelfAttention:
    """Build one token per piece feature block, then apply self-attention over pieces."""

    def __init__(self,ips,mps,ops,piece_feature_indices = None):
        if piece_feature_indices is None:
            piece_feature_indices = ops
            ops = mps
            mps = ops
        self.ips = int(ips)
        self.mps = int(mps)
        self.ops = int(ops)
        self.W = np.random.randn(self.mps,self.ips).astype('f') / math.sqrt(max(self.ips,1))
        self.B = np.zeros(self.ops,dtype = 'f')
        self.WQ = np.random.randn(self.mps,self.mps).astype('f') * (0.1 / math.sqrt(max(self.mps,1)))
        self.WK = np.random.randn(self.mps,self.mps).astype('f') * (0.1 / math.sqrt(max(self.mps,1)))
        self.WV = np.random.randn(self.ops,self.mps).astype('f') / math.sqrt(max(self.mps,1))
        self.piece_feature_indices = [np.asarray(indices,dtype = int) for indices in piece_feature_indices]
        self.scale = 1.0 / math.sqrt(max(self.mps,1))
        self.x = None
        self.tokens = None
        self.Q = None
        self.K = None
        self.V = None
        self.M = None
        self.attended_sum = None
        self.out = None
        self.backward_chunk_size = 32
        self.forward_chunk_size = 256
        self.dW = np.zeros_like(self.W,dtype = 'f')
        self.dB = np.zeros_like(self.B,dtype = 'f')
        self.dWQ = np.zeros_like(self.WQ,dtype = 'f')
        self.dWK = np.zeros_like(self.WK,dtype = 'f')
        self.dWV = np.zeros_like(self.WV,dtype = 'f')

    def forward(self,x,cache = True):
        if cache:
            self.x = x
            self.tokens = self._piece_tokens(x)
            self.Q,self.K,self.V,self.M,self.out,self.attended_sum = self._forward_from_tokens(self.tokens)
            return self.out
        return self._forward_no_cache(x)

    def _forward_no_cache(self,x):
        self.x = None
        self.tokens = None
        self.Q = None
        self.K = None
        self.V = None
        self.M = None
        self.attended_sum = None
        self.out = None
        out = np.zeros((self.ops,x.shape[1]),dtype = 'f')
        chunk_size = max(1,min(int(self.forward_chunk_size),x.shape[1]))
        for start in range(0,x.shape[1],chunk_size):
            end = min(start + chunk_size,x.shape[1])
            _,_,_,_,chunk_out,_ = self._forward_from_tokens(self._piece_tokens(x[:,start:end]))
            out[:,start:end] = chunk_out
        return out

    def _forward_from_tokens(self,tokens):
        q = np.einsum('od,pdb->pob',self.WQ,tokens)
        k = np.einsum('od,pdb->pob',self.WK,tokens)
        v = tokens
        score = self.scale * np.einsum('pdb,qdb->bpq',q,k)
        attention = softmax(score)
        attended = np.einsum('bpq,qdb->pdb',attention,v)
        attended_sum = np.sum(attended,axis = 0)
        out = self.WV @ attended_sum + self.B.reshape(-1,1)
        return q,k,v,attention,out.astype('f'),attended_sum.astype('f')

    def _piece_tokens(self,x):
        token_count = len(self.piece_feature_indices)
        tokens = np.zeros((token_count,self.W.shape[0],x.shape[1]),dtype = 'f')
        for piece_index, feature_indices in enumerate(self.piece_feature_indices):
            if feature_indices.size == 0:
                continue
            tokens[piece_index] = self.W[:,feature_indices] @ x[feature_indices]
        return tokens

    def backward(self,dO):
        dx = np.zeros_like(self.x,dtype = 'f')
        self.dW = np.zeros_like(self.W,dtype = 'f')
        self.dWQ = np.zeros_like(self.WQ,dtype = 'f')
        self.dWK = np.zeros_like(self.WK,dtype = 'f')
        self.dWV = np.zeros_like(self.WV,dtype = 'f')
        batch_size = dO.shape[1]
        chunk_size = max(1,min(int(self.backward_chunk_size),batch_size))
        for start in range(0,batch_size,chunk_size):
            end = min(start + chunk_size,batch_size)
            dx[:,start:end] += self._backward_chunk(dO[:,start:end],start,end)
        self.dB = np.sum(dO,axis = 1)
        return dx

    def _backward_chunk(self,dO,start,end):
        tokens = self.tokens[:,:,start:end]
        q = self.Q[:,:,start:end]
        k = self.K[:,:,start:end]
        v = self.V[:,:,start:end]
        attention = self.M[start:end]
        attended_sum = self.attended_sum[:,start:end]
        self.dWV += dO @ attended_sum.T
        dA = self.WV.T @ dO
        dS_base = np.einsum('db,qdb->bq',dA,v)
        dS = np.broadcast_to(dS_base.reshape(dS_base.shape[0],1,dS_base.shape[1]),attention.shape).copy()
        row_dot = np.sum(dS * attention,axis = -1,keepdims = True)
        dS -= row_dot
        dS *= attention
        dV = np.einsum('bq,db->qdb',np.sum(attention,axis = 1),dA)
        k_batch = np.ascontiguousarray(np.transpose(k,(2,0,1)))
        q_batch = np.ascontiguousarray(np.transpose(q,(2,0,1)))
        dQ = self.scale * np.matmul(dS,k_batch).transpose(1,2,0)
        dK = self.scale * np.matmul(np.transpose(dS,(0,2,1)),q_batch).transpose(1,2,0)

        token_matrix = tokens.transpose(1,0,2).reshape(tokens.shape[1],-1)
        self.dWQ += dQ.transpose(1,0,2).reshape(dQ.shape[1],-1) @ token_matrix.T
        self.dWK += dK.transpose(1,0,2).reshape(dK.shape[1],-1) @ token_matrix.T

        d_tokens = self.WQ.T @ dQ.transpose(1,0,2).reshape(dQ.shape[1],-1)
        d_tokens += self.WK.T @ dK.transpose(1,0,2).reshape(dK.shape[1],-1)
        d_tokens += dV.transpose(1,0,2).reshape(dV.shape[1],-1)
        d_tokens = d_tokens.reshape(tokens.shape[1],tokens.shape[0],tokens.shape[2]).transpose(1,0,2)

        dx = np.zeros((self.x.shape[0],end - start),dtype = 'f')
        for piece_index, feature_indices in enumerate(self.piece_feature_indices):
            if feature_indices.size == 0:
                continue
            self.dW[:,feature_indices] += d_tokens[piece_index] @ self.x[feature_indices,start:end].T
            dx[feature_indices] += self.W[:,feature_indices].T @ d_tokens[piece_index]
        return dx

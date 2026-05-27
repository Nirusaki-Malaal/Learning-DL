import numpy as np
import signal, sys
import os, functools
from PIL import Image

class Layer:
    def __init__(self, n_in, n_out,alpha=1e-3):
        ## see i am going to perform an affine transformation here into a linear combination of systems 
        self.n_in = n_in ## no. of features
        self.n_out = n_out              ## current layer ke neurons hai yeh
        rng = np.random.default_rng() ## random variable generator object using any probablity distribution i mention
        std = np.sqrt(2 / self.n_in) # He init initialization
        self.W = rng.normal(0, std, (self.n_out, self.n_in)) ## (mean , standard_deviation, (shape tuple)) gaussian distribution used
        self.b = np.zeros((self.n_out,1))
        # for back propagation later
        self.X = None
        self.Z = None
        self.alpha = alpha
        self.A = None
        
    def forward(self, X, activation:str): ## forward propagation
        self.X = X 
        self.Z = self.W @ self.X + self.b ## affine transformation
        if activation == "relu":
            self.A = self.relu(self.Z)
        elif activation == "softmax":
            self.A = self.softmax(self.Z)
          
        return self.A    
    
    def backward(self,dz): ## gradient flows backward
        m = self.X.shape[1]
        dw = (dz @ self.X.T) / (m)
        db = (np.sum(dz, axis=1, keepdims=True)) / m
        dA = self.W.T @ dz
        self.W-=self.alpha * dw
        self.b-=self.alpha * db
        return dA
    
    def relu(self,Z):
        return np.maximum(0,Z)

    def softmax(self,Z):
        exp_z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return  exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def compute_loss(self, A, Y):
        return -np.sum(Y * np.log(A + 1e-8)) / Y.shape[1]

def exitfunc(signum, frame, layers):
    save_data(layers)
    sys.exit(0)


def load_image(path):
    img = Image.open(path).convert("L")
    img = np.array(img)
    img = (img > 100).astype(np.uint8) * 255
    img = 255 - img
    coords = np.argwhere(img > 0)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1+1, x0:x1+1]
    img = Image.fromarray(img).resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(784, 1)

    return img

def full_forward(X, layers):
        A = X
        for j, layer in enumerate(layers):
            activation = "softmax" if j == len(layers) - 1 else "relu"
            A = layer.forward(A, activation)
        return A

def save_data(layers, path="model.npz"):
    data = {}
    for i, layer in enumerate(layers):
        data[f"W{i}"] = layer.W
        data[f"b{i}"] = layer.b
    np.savez(path, **data)
    print(f"Model saved {path}")

def load_data(layers, path="model.npz"):
    checkpoint = np.load(path)
    for i, layer in enumerate(layers):
        layer.W = checkpoint[f"W{i}"]
        layer.b = checkpoint[f"b{i}"]
    print(f"Model loaded {path}")
    return layers

def predict(layers,path):
    X = load_image(path)
    A = full_forward(X, layers)
    return np.argmax(A, axis=0)

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.T / 255.0 # normalization to a range of (0,1)
    labels = mnist.target.astype(int)
    Y = np.zeros((10, 70000))
    Y[labels, np.arange(70000)] = 1  
    k = 10 # classes
    layers = [Layer(X.shape[0],128), Layer(128,64), Layer(64,32), Layer(32,k)]

    if os.path.exists("model.npz"):
        layers = load_data(layers , "model.npz")
        print("Resuming from checkpoint")
    signal.signal(signal.SIGINT, functools.partial(exitfunc, layers=layers))
    
    a = input("Type 'train' or 'predict': ").strip().lower()

    if a == "predict":
        path = input("enter filename")
        if os.path.exists(path):
            print(predict(layers,path))
            exit()
        else:
            print("enter correct path")
            exit()

    for epoch in range(5000):
            batch_size = 128
            m = X.shape[1]
            perm = np.random.permutation(m)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[:, perm]
            for i in range(0, m, batch_size):
                Xb = X_shuffled[:, i:i+batch_size]
                Yb = Y_shuffled[:, i:i+batch_size]
                A = Xb
                for j in range(len(layers)):
                    if j == len(layers) - 1:
                        A = layers[j].forward(A, "softmax")
                    else:
                        A = layers[j].forward(A, "relu")
                dz = A - Yb
                dA = dz
                for j in reversed(range(len(layers))):
                    if j != len(layers) - 1:
                        dA = dA * (layers[j].Z > 0)
                    dA = layers[j].backward(dA)
            if epoch % 100 == 0:
                save_data(layers, "model.npz")
                A_full = full_forward(X, layers)
                loss = layers[-1].compute_loss(A_full, Y)
                acc = np.mean(np.argmax(A_full, axis=0) == labels) * 100
                print(f"epoch {epoch:4d}  loss {loss:.4f}  acc {acc:.2f}%")
        
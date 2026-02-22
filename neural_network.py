import numpy as np
import pandas as pd
import kagglehub
import os

# Color to cost mapping from instructions
COLOR_COSTS = {
    'Green': 3,
    'Blue': 10,
    'Grey': 1,
    'Yellow': 5
}
DEFAULT_COST = 1

def get_dataset():
    path = kagglehub.dataset_download("jasminebach/basiccolornames")
    df = pd.read_csv(os.path.join(path, "final_data_colors.csv"))
    # The columns are ['red', 'green', 'blue', 'label']
    return df

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases (Xavier initialization)
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        # Subtract max for numerical stability
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, Y, learning_rate):
        m = X.shape[0]
        
        # dZ2 is A2 - Y for cross entropy loss with softmax
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * self.A1.T.dot(dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = dZ2.dot(self.W2.T) * self.relu_deriv(self.Z1)
        dW1 = (1 / m) * X.T.dot(dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # SGD / Batch GD update
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X_train, Y_train, epochs=2000, learning_rate=0.1, batch_size=32):
        m = X_train.shape[0]
        indices = np.arange(m)
        for i in range(epochs):
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            
            for j in range(0, m, batch_size):
                X_batch = X_shuffled[j:j+batch_size]
                Y_batch = Y_shuffled[j:j+batch_size]
                
                self.forward(X_batch)
                self.backward(X_batch, Y_batch, learning_rate)
            
            if (i+1) % 100 == 0:
                predictions = self.predict(X_train)
                acc = np.mean(predictions == np.argmax(Y_train, axis=1))
                print(f"Epoch {i+1}/{epochs}: Train Accuracy = {acc:.4f}")

    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=1)

    def save_weights(self, filename='mlp_weights.npz'):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_weights(self, filename='mlp_weights.npz'):
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

def prepare_data(df):
    X = df[['red', 'green', 'blue']].values / 255.0  # Normalize RGB to 0-1
    labels = df['label'].values
    unique_labels = np.unique(labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    Y_idx = np.array([label_to_idx[l] for l in labels])
    # One-hot encoding
    Y = np.zeros((labels.size, len(unique_labels)))
    Y[np.arange(labels.size), Y_idx] = 1
    
    return X, Y, Y_idx, label_to_idx, idx_to_label

def train_and_save():
    print("Loading dataset...")
    df = get_dataset()
    X, Y, Y_idx, label_to_idx, idx_to_label = prepare_data(df)
    
    # Save the label mapping
    np.save('idx_to_label.npy', idx_to_label)
    
    # 80-20 Train-Test split
    np.random.seed(123)
    m = X.shape[0]
    indices = np.random.permutation(m)
    train_size = int(0.8 * m)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test_idx = X[test_idx], Y_idx[test_idx]
    
    # Architecture: 3 inputs, 32 hidden neurons, 11 outputs
    mlp = MLP(input_size=3, hidden_size=64, output_size=len(label_to_idx))
    
    print(f"Training MLP... (Train set: {train_size}, Test set: {m - train_size})")
    mlp.train(X_train, Y_train, epochs=800, learning_rate=0.4, batch_size=128)
    
    predictions = mlp.predict(X_test)
    test_acc = np.mean(predictions == Y_test_idx)
    print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    mlp.save_weights()
    print("Model weights saved to mlp_weights.npz")
    print("Label mappings saved to idx_to_label.npy")

def load_inference_model():
    idx_to_label = np.load('idx_to_label.npy', allow_pickle=True).item()
    mlp = MLP(input_size=3, hidden_size=64, output_size=len(idx_to_label))
    mlp.load_weights('mlp_weights.npz')
    return mlp, idx_to_label

def predict_color(rgb, mlp, idx_to_label):
    X = np.array(rgb).reshape(1, -1) / 255.0
    pred_idx = mlp.predict(X)[0]
    return idx_to_label[pred_idx]

def get_cost_for_color(color_name):
    # Retrieve cost mapped to color, or fallback to default
    return COLOR_COSTS.get(color_name, DEFAULT_COST)

if __name__ == "__main__":
    train_and_save()

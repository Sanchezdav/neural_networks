# Neural Networks Learning Repository

A hands-on collection of neural network implementations and exercises built from scratch to understand the fundamentals of deep learning.

## Overview

This repository is dedicated to learning neural networks through practical implementation. Each example is designed to be clear and self-contained, focusing on understanding core concepts by building everything from first principles without relying on machine learning frameworks.

## Purpose

The goal is to deeply understand how neural networks work by:
- Implementing the mathematics from scratch
- Visualizing each step of the learning process
- Experimenting with hyperparameters and architectures
- Building intuition about gradient descent, backpropagation, and optimization

## Core Concepts

### Forward Propagation
The process of passing input data through the network to generate a prediction.

```
Input → Weighted Sum → Activation → Hidden Layer → Output → Prediction
```

Each layer transforms the input using learned weights and biases, applying activation functions to introduce non-linearity.

### Backpropagation
The algorithm for computing gradients by propagating errors backward through the network using the chain rule.

```
Loss Function → Output Error → Hidden Layer Error → Weight Gradients
```

This tells us how much each weight contributed to the overall error.

### Gradient Descent
An optimization algorithm that iteratively adjusts weights to minimize the loss function.

```ruby
new_weight = old_weight - (learning_rate × gradient)
```

The learning rate controls the step size, and the gradient points in the direction of steepest increase (we go opposite to decrease loss).

### Activation Functions

**ReLU (Rectified Linear Unit):**
```ruby
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```
- Simple and computationally efficient
- Helps prevent vanishing gradient problem
- Most common for hidden layers

**Sigmoid:**
```ruby
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) × (1 - f(x))
```
- Outputs values between 0 and 1
- Useful for binary classification
- Can suffer from vanishing gradients

**Tanh:**
```ruby
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
f'(x) = 1 - f(x)²
```
- Outputs values between -1 and 1
- Zero-centered (unlike sigmoid)
- Similar gradient issues as sigmoid

### Loss Functions

**Mean Squared Error (MSE):**
```ruby
loss = (prediction - actual)²
```
- Used for regression problems
- Differentiable and smooth
- Penalizes larger errors quadratically

**Cross-Entropy Loss:**
```ruby
loss = -[y × log(ŷ) + (1 - y) × log(1 - ŷ)]
```
- Used for classification problems
- Measures divergence between probability distributions
- Works well with sigmoid/softmax outputs

### Network Architecture Components

**Weights (W):**
- Learned parameters that determine connection strength between neurons
- Initialized randomly to break symmetry

**Biases (b):**
- Learned parameters that allow shifting the activation function
- Help the network fit the data better

**Layers:**
- **Input Layer:** Raw features fed into the network
- **Hidden Layers:** Intermediate representations that extract features
- **Output Layer:** Final predictions

## Key Learning Topics

### 1. The Learning Process

Neural networks learn through an iterative process:

1. **Initialize** weights randomly
2. **Forward Pass:** Make predictions with current weights
3. **Calculate Loss:** Measure how wrong the predictions are
4. **Backward Pass:** Compute gradients via backpropagation
5. **Update Weights:** Adjust weights using gradient descent
6. **Repeat:** Continue until loss is minimized

### 2. Hyperparameters

**Learning Rate:**
- Controls how much weights change each iteration
- Too large: unstable training, overshooting
- Too small: slow convergence, stuck in local minima
- Typical range: 0.001 to 0.0001

**Epochs:**
- Number of complete passes through the training data
- More epochs = more learning opportunities
- Too many can lead to overfitting

**Batch Size:**
- Number of samples processed before updating weights
- Full batch: stable but slow
- Mini-batch: balance of speed and stability
- Single sample (SGD): noisy but can escape local minima

**Network Architecture:**
- Number of hidden layers (depth)
- Number of neurons per layer (width)
- Choice of activation functions

### 3. Common Challenges

**Vanishing Gradients:**
- Gradients become extremely small in deep networks
- Causes: sigmoid/tanh saturation
- Solutions: ReLU, residual connections, careful initialization

**Exploding Gradients:**
- Gradients become extremely large
- Causes: poor initialization, high learning rates
- Solutions: gradient clipping, lower learning rate

**Overfitting:**
- Network memorizes training data instead of learning patterns
- Signs: training loss decreases but validation loss increases
- Solutions: more data, regularization (L1/L2), dropout, early stopping

**Underfitting:**
- Network is too simple to capture data patterns
- Signs: both training and validation loss remain high
- Solutions: deeper network, more neurons, train longer

### 4. Gradient Calculation (Chain Rule)

For a simple network: Input → Hidden → Output

```
∂Loss/∂w_hidden = ∂Loss/∂output × ∂output/∂hidden × ∂hidden/∂w_hidden
```

This is the **chain rule** in action. Each layer's gradient depends on all downstream layers.

### 5. Why Neural Networks Work

**Universal Approximation Theorem:**
- A neural network with a single hidden layer can approximate any continuous function
- Given enough neurons and proper training

**Feature Learning:**
- Networks automatically learn hierarchical representations
- Early layers: simple features (edges, colors)
- Deep layers: complex features (objects, concepts)

**Non-linearity:**
- Activation functions allow networks to learn non-linear patterns
- Without activation, multiple layers collapse to single linear transformation

## Experimentation Ideas

### Learning Rate Exploration
```ruby
# Try these values and observe:
learning_rate = 0.1      # What happens?
learning_rate = 0.001    # What happens?
learning_rate = 0.00001  # What happens?
```

### Architecture Variations
```ruby
# Experiment with:
- Different number of hidden neurons
- Multiple hidden layers
- Different activation functions
- Skip connections
```

### Initialization Strategies
```ruby
# Compare:
- Random small values: rand(-0.5, 0.5)
- Xavier initialization: rand(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
- He initialization: rand(0, √(2/n_in))
- All zeros (observe the symmetry problem!)
```

### Monitoring Training
```ruby
# Track and plot:
- Training loss over epochs
- Weight magnitudes
- Gradient magnitudes
- Prediction accuracy
```

## Mathematical Foundations

### Required Background

**Calculus:**
- Derivatives and partial derivatives
- Chain rule for composition of functions
- Gradient descent optimization

**Linear Algebra:**
- Matrix and vector operations
- Dot products
- Matrix multiplication for efficient computation

**Probability & Statistics:**
- Loss functions as negative log likelihood
- Expectation and variance
- Regularization as prior beliefs

## Best Practices

### For Learning
1. **Start simple:** Small networks, small datasets
2. **Understand first:** Know what each line does before moving on
3. **Visualize:** Plot loss curves, weight distributions, activations
4. **Debug systematically:** Check gradients, verify forward pass, test components
5. **Experiment:** Change one thing at a time and observe effects

### For Implementation
1. **Vectorize operations:** Use matrix operations instead of loops
2. **Normalize inputs:** Scale features to similar ranges
3. **Monitor gradients:** Check for vanishing/exploding gradients
4. **Use validation set:** Separate data for tuning hyperparameters
5. **Save checkpoints:** Keep best model during training

## Common Debugging Techniques

### Gradient Checking
Compare numerical gradients with analytical gradients:
```ruby
numerical_gradient = (loss(w + ε) - loss(w - ε)) / (2ε)
analytical_gradient = backprop_gradient
assert (numerical_gradient - analytical_gradient).abs < 1e-7
```

### Overfit a Small Sample
Try to perfectly fit 1-3 examples:
- If you can't, there's a bug
- If you can, the network has capacity to learn

### Check Loss at Initialization
For classification, initial loss should be near -log(1/num_classes)

### Visualize Predictions
Plot predictions vs actual values to spot patterns

## Resources for Further Learning

### Concepts to Explore Next
- Convolutional Neural Networks (CNNs) for images
- Recurrent Neural Networks (RNNs) for sequences
- Attention mechanisms and Transformers
- Batch normalization and layer normalization
- Advanced optimizers (Adam, RMSprop, AdaGrad)
- Regularization techniques (dropout, L1/L2)
- Transfer learning and pre-training

### Mathematical Deep Dives
- Automatic differentiation
- Computational graphs
- Optimization theory
- Information theory and entropy

## Tools & Technologies

**Current Approach:**
- Language: Ruby (or any programming language)
- Philosophy: From-scratch implementation
- Focus: Understanding over performance

**Next Steps:**
- Frameworks: PyTorch, TensorFlow, JAX
- Tools: Jupyter notebooks, TensorBoard
- Libraries: NumPy, Pandas for data handling

## Philosophy

> "What I cannot create, I do not understand." - Richard Feynman

This repository embraces building from scratch to develop deep understanding. While production systems use optimized frameworks, implementing the fundamentals yourself reveals insights that can't be gained any other way.

The goal isn't to build the fastest or most accurate model, but to understand **why** neural networks work and **how** to think about them.

---

## Documentation in This Repository

Check the included documentation files for detailed walkthroughs, mathematical derivations, and step-by-step examples with actual calculations.

**Remember:** Neural networks are powerful because they can learn from data, but understanding the mechanisms behind that learning is what separates practitioners from experts.

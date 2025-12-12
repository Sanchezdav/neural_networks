# frozen_string_literal: true

# Neural Network: Predicting Student Test Scores
# Architecture: 2 inputs → 2 hidden neurons (ReLU) → 1 output
# Optimizer: SGD (Stochastic Gradient Descent)

# Training data: [hours_studied, hours_slept] → exam_score
training_data = [
  [[5, 8], 85],
  [[2, 6], 60],
  [[8, 7], 95]
]

# Hyperparameters
# Learning rate controls how big the steps are when updating weights
# Epochs define how many times we go through the entire training dataset
learning_rate = 0.0001
epochs = 1000

# Initialize random weights and biases
w1 = [[0.4, 0.6], [0.3, -0.2]] # Input → Hidden (2x2)
w2 = [[0.8], [0.5]]              # Hidden → Output (2x1)
b1 = [0.5, -0.1]                 # Hidden biases
b2 = [10.0]                      # Output bias

# ReLU activation function
def relu(sum)
  [0, sum].max
end

# Training loop
puts 'Starting training...'
puts '-' * 50

# rubocop:disable Metrics/BlockLength
epochs.times do |epoch|
  total_loss = 0

  training_data.each do |input, actual|
    # === FORWARD PASS ===

    # Hidden layer calculations
    ## Weighted sum (also called linear combination or pre-activation)
    z1 = (input[0] * w1[0][0]) + (input[1] * w1[1][0]) + b1[0]
    h1 = relu(z1)

    z2 = (input[0] * w1[0][1]) + (input[1] * w1[1][1]) + b1[1]
    h2 = relu(z2)

    # Output layer
    prediction = (h1 * w2[0][0]) + (h2 * w2[1][0]) + b2[0]

    # === CALCULATE LOSS ===

    # Output layer error
    error = prediction - actual

    loss = error**2
    total_loss += loss

    # === BACKPROPAGATION ===

    # Gradients for w2 (Hidden → Output)
    grad_hidden1_to_output = error * h1
    grad_hidden2_to_output = error * h2
    grad_output_bias = error

    # Propagate error back to hidden layer
    error_h1 = error * w2[0][0] * (z1.positive? ? 1 : 0) # ReLU derivative
    error_h2 = error * w2[1][0] * (z2.positive? ? 1 : 0)

    # Gradients for w1 (Input → Hidden)
    grad_studied_to_hidden1 = error_h1 * input[0]
    grad_slept_to_hidden1 = error_h1 * input[1]
    grad_studied_to_hidden2 = error_h2 * input[0]
    grad_slept_to_hidden2 = error_h2 * input[1]
    grad_hidden1_bias = error_h1
    grad_hidden2_bias = error_h2

    # === UPDATE WEIGHTS (SGD) ===

    # Update w2
    w2[0][0] -= learning_rate * grad_hidden1_to_output
    w2[1][0] -= learning_rate * grad_hidden2_to_output
    b2[0] -= learning_rate * grad_output_bias

    # Update w1
    w1[0][0] -= learning_rate * grad_studied_to_hidden1
    w1[1][0] -= learning_rate * grad_slept_to_hidden1
    w1[0][1] -= learning_rate * grad_studied_to_hidden2
    w1[1][1] -= learning_rate * grad_slept_to_hidden2
    b1[0] -= learning_rate * grad_hidden1_bias
    b1[1] -= learning_rate * grad_hidden2_bias
  end

  # Calculate average loss
  avg_loss = total_loss / training_data.length

  # Print progress every 100 epochs
  puts "Epoch #{epoch + 1}: Avg Loss = #{avg_loss.round(2)}" if ((epoch + 1) % 50).zero?
end
# rubocop:enable Metrics/BlockLength

puts '-' * 50
puts 'Training complete!'
puts

# === FINAL WEIGHTS ===
puts 'Final weights:'
puts 'w1 (Input → Hidden):'
puts "  Hours studied → H1: #{w1[0][0].round(3)}, H2: #{w1[0][1].round(3)}"
puts "  Hours slept → H1: #{w1[1][0].round(3)}, H2: #{w1[1][1].round(3)}"
puts 'w2 (Hidden → Output):'
puts "  H1 → Score: #{w2[0][0].round(3)}"
puts "  H2 → Score: #{w2[1][0].round(3)}"
puts "Biases: b1 = [#{b1[0].round(3)}, #{b1[1].round(3)}], b2 = [#{b2[0].round(3)}]"
puts

# === TEST THE NETWORK ===
puts 'Testing the trained network:'
puts '-' * 50

test_cases = [
  [5, 8],  # Original training example
  [2, 6],  # Original training example
  [8, 7],  # Original training example
  [6, 7],  # New student
  [3, 5],  # Low effort
  [9, 8]   # High effort
]

test_cases.each do |input|
  # Forward pass
  z1 = (input[0] * w1[0][0]) + (input[1] * w1[1][0]) + b1[0]
  h1 = relu(z1)

  z2 = (input[0] * w1[0][1]) + (input[1] * w1[1][1]) + b1[1]
  h2 = relu(z2)

  prediction = (h1 * w2[0][0]) + (h2 * w2[1][0]) + b2[0]

  puts "Input: [studied: #{input[0]}h, slept: #{input[1]}h] → Predicted score: #{prediction.round(1)}"
end

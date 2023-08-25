const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const util = require('util');

// Set the random seed for reproducibility
tf.setSeed(42);

// Generate synthetic data
const num_samples = 1000;
const max_length = 500;  // Maximum length of the essays
const vocab_size = 10000;  // Vocabulary size for the tokenizer

const essays = [];
const scores = [];

for (let i = 0; i < num_samples; i++) {
    const essay_length = tf.util.randUniform(100, max_length, dtype='int32');
    const essay = Array.from({ length: essay_length }, () => String(Math.floor(tf.util.randUniform(0, vocab_size, dtype='int32'))));
    const score = tf.util.randUniform(0, 10, dtype='int32');
    essays.push(essay.join(' '));
    scores.push(score);
}

// Convert scores to a tensor
const scoresTensor = tf.tensor(scores, [num_samples]);

const { TextTokenizer } = require('@tensorflow/tfjs-node/dist/tokenizer');
const { encodeString, padTensor } = require('@tensorflow/tfjs-node/dist/text/tokenizer_utils');

// Tokenize the text
const tokenizer = new TextTokenizer(vocab_size);
tokenizer.update(essays);

// Convert text to sequences of integers
const sequences = tokenizer.encode(essays);

// Pad sequences to a fixed length
const max_length = 500;
const data = padTensor(sequences, max_length);

// Split the data into training and testing sets
const train_ratio = 0.8;
const train_samples = Math.floor(num_samples * train_ratio);

const x_train = data.slice([0, 0], [train_samples, max_length]);
const y_train = scoresTensor.slice([0], [train_samples]);

const x_test = data.slice([train_samples, 0], [num_samples - train_samples, max_length]);
const y_test = scoresTensor.slice([train_samples]);

// Build and train the LSTM model
const embedding_dim = 100;
const hidden_units = 64;

const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: vocab_size, outputDim: embedding_dim, inputLength: max_length }));
model.add(tf.layers.lstm({ units: hidden_units }));
model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });
model.fit(x_train, y_train, { batchSize: 32, epochs: 10, validationData: [x_test, y_test] })
    .then(async () => {
        // Evaluate the trained model
        const evalResult = model.evaluate(x_test, y_test);
        console.log(`Test Loss: ${evalResult}`);
    })
    .catch((error) => {
        console.error('Error:', error);
    });

# 05_Basic Fine-tuning of BERT for NLP Tasks

First read About_Bert.md if you want to know more

Use gpu runtime while running the notebook

## BERT and Fine-tuning:
BERT (Bidirectional Encoder Representations from Transformers) is a powerful pre-trained language model that has revolutionized natural language processing tasks. It is based on the Transformer architecture and is trained on a massive corpus of text data in an unsupervised manner, learning rich contextual representations of language.

![bert structure](https://media.geeksforgeeks.org/wp-content/uploads/20200422005041/NextSentencePrediction.jpg)

In this code, the DistilBERT variant is used, which is a distilled version of the original BERT model, designed to be smaller and more efficient while preserving most of the performance.

Fine-tuning is the process of taking a pre-trained language model like BERT and further training it on a specific task and dataset. This allows the model to adapt to the task at hand, leveraging the knowledge and representations it has already learned from the pre-training phase.

In the context of this code, the DistilBERT model is fine-tuned on the sentiment analysis task using the provided restaurant review dataset. During fine-tuning, the model's parameters are adjusted to better capture the nuances and patterns specific to the sentiment analysis task, while still benefiting from the general language understanding acquired during pre-training.

The advantages of fine-tuning include:

1. **Transfer Learning**: By starting from a pre-trained model, the fine-tuning process can leverage the rich language representations learned from a vast amount of data, reducing the need for extensive task-specific training data.

2. **Improved Performance**: Fine-tuning allows the model to specialize in the target task, often leading to better performance compared to training a model from scratch on the same task-specific data.

3. **Faster Convergence**: Since the pre-trained model already captures general language patterns, fine-tuning requires fewer training iterations to adapt to the target task, resulting in faster convergence and reduced training time.

In summary, the provided code demonstrates the end-to-end process of fine-tuning a pre-trained language model (DistilBERT) for sentiment analysis on restaurant reviews. It showcases data preparation, model loading, training configuration, fine-tuning using the Hugging Face Transformers library, and inference for sentiment prediction.

The code demonstrates a practical implementation of fine-tuning a pre-trained language model, specifically DistilBERT, for sentiment analysis on restaurant reviews. Here's a detailed explanation of the code:

1. **Data Preparation**:
   - The code starts by importing necessary libraries such as pandas, numpy, seaborn, and matplotlib.
   - It reads a CSV file containing restaurant reviews and ratings.
   - The dataset is filtered to include only reviews with ratings of 1 (negative) or 5 (positive).
   - A function `sample_by_sentiment` is defined to sample an equal number of reviews from each sentiment class (1 and 5), ensuring a balanced dataset.
   - The 'Rating' column is replaced with a binary 'sentiment' column (0 for negative and 1 for positive).

2. **Train-Test Split**:
   - The dataset is split into training and test sets using `train_test_split` from scikit-learn.

3. **Tokenization and Dataset Creation**:
   - The DistilBERT tokenizer (`DistilBertTokenizerFast`) is loaded from the pre-trained model.
   - The reviews in the training and test sets are tokenized using the tokenizer, with truncation and padding applied.
   - A custom PyTorch Dataset class `data` is defined to handle the tokenized data and associated labels.
   - Instances of the `data` class are created for the training and test sets.

4. **Model and Training Configuration**:
   - The DistilBERT model (`DistilBertForSequenceClassification`) is loaded from the pre-trained model.
   - Training arguments (`TrainingArguments`) are defined, which include hyperparameters such as the number of epochs, batch size, learning rate, warmup steps, and logging/saving configurations.
   - The Trainer object from the Hugging Face Transformers library is initialized with the model, training arguments, and the training and test datasets.

5. **Fine-tuning and Accelerator**:
   - The Accelerator library is imported and initialized, which handles efficient training on various hardware configurations (CPU, GPU, or TPU).
   - The `trainer.train()` method is called to start the fine-tuning process.

6. **Inference and Prediction**:
   - A `predict_sentiment` function is defined, which takes the fine-tuned model, tokenizer, text input, and device as arguments.
   - The text is tokenized, and the model's output logits are obtained.
   - The softmax function is applied to the logits to obtain prediction probabilities.
   - The prediction (positive or negative sentiment) and its probability are printed.
   - Two example texts are provided to demonstrate the sentiment prediction using the fine-tuned model.

## Some more information 

1. **Tokenizer**:
   - The tokenizer is responsible for converting the raw text input into a format that the BERT model can understand.
   - In this code, `DistilBertTokenizerFast` is used, which is a fast and efficient tokenizer specifically designed for the DistilBERT model.
   - The tokenizer splits the input text into smaller units called tokens, which can be words, subwords, or even individual characters, depending on the tokenization strategy.
   - The tokenizer also handles operations like truncation (limiting the input length) and padding (adding special tokens to ensure a fixed input length).
   - The tokenizer maps each token to a numerical value (token ID) based on its pre-trained vocabulary.

2. **Encoding and Decoding**:
   - Encoding refers to the process of converting the input text into a numerical representation that the model can process.
   - The tokenizer performs encoding by splitting the text into tokens and converting them into token IDs.
   - The resulting token IDs, along with other special tokens (like start and end tokens), form the input to the BERT model.
   - Decoding is the reverse process, where the model's output (a sequence of token IDs) is converted back into human-readable text by the tokenizer.

3. **DistilBERT Model**:
   - DistilBERT is a smaller and more efficient variant of the original BERT model, obtained through a process called distillation.
   - It aims to capture most of the performance of the larger BERT model while being more efficient in terms of computational resources and memory requirements.
   - In this code, `DistilBertForSequenceClassification` is used, which is a pre-trained DistilBERT model specifically designed for sequence classification tasks like sentiment analysis.
   - The model takes the encoded input sequences and processes them through multiple layers of self-attention and feed-forward networks to capture the contextual relationships between tokens.
   - The final output of the model is a classification score (logits) for each possible class (in this case, positive or negative sentiment).

4. **Trainer**:
   - The `Trainer` class from the Hugging Face Transformers library is a high-level abstraction that handles the training loop, evaluation, and model checkpointing.
   - It simplifies the training process by providing a unified interface for various models and tasks.
   - The `Trainer` is initialized with the model, training arguments, and the training and evaluation datasets.
   - It automates tasks like data loading, gradient computation, optimization, and evaluation metrics calculation.

5. **Accelerator**:
   - The `Accelerator` class from the Hugging Face Accelerate library is a tool for efficient training and inference on various hardware configurations (CPU, GPU, or TPU).
   - It automatically handles distributed training, mixed precision, and other optimization techniques to accelerate the training process.
   - In the provided code, the `Accelerator` is initialized before training, which allows the `Trainer` to leverage the available hardware resources effectively.

6. **Training Arguments**:
   - The `TrainingArguments` class is used to configure various hyperparameters and settings for the training process.
   - Some key arguments include:
     - `num_train_epochs`: The number of epochs (full passes through the training data) to perform.
     - `per_device_train_batch_size` and `per_device_eval_batch_size`: The batch sizes for training and evaluation, respectively.
     - `learning_rate`: The initial learning rate for the optimizer.
     - `warmup_steps`: The number of warmup steps for the learning rate scheduler.
     - `weight_decay`: The weight decay coefficient for regularization.
     - `logging_steps` and `save_steps`: The frequency of logging and saving model checkpoints during training.
     - `evaluation_strategy`: The strategy for evaluating the model on the validation set during training.

7. **Other Ways to Fine-tune BERT**:
   - While the provided code uses the Hugging Face Transformers library, it is also possible to fine-tune BERT models using other deep learning frameworks like PyTorch or TensorFlow.
   - The general process involves loading the pre-trained BERT model weights, adding a task-specific classification layer, and training the model on the target dataset using appropriate loss functions and optimization techniques.
   - Different fine-tuning strategies can be employed, such as layer-wise fine-tuning (only fine-tuning the top layers) or gradual unfreezing (gradually unfreezing and fine-tuning layers from the bottom up).
   - Techniques like data augmentation, transfer learning from other tasks, and hyperparameter tuning can be explored to improve the fine-tuning performance.

In summary, this code demonstrates a complete pipeline for fine-tuning the DistilBERT model for sentiment analysis. It involves tokenization, encoding, loading the pre-trained model, configuring training arguments, and leveraging the Hugging Face Transformers and Accelerate libraries for efficient training and inference.
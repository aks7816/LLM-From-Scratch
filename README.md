## Implements a GPT (Generative Pre-trained Transformer) language model from scratch using PyTorch
### This is a simplified but functional implementation of the GPT architecture and follows a transformer architecture


Overall Purpose


-This is a basic GPT (AI text generator) built from scratch
-Two files: one for training, one for chatting
-Uses PyTorch to handle the AI math


Main Building Blocks


-Attention Heads: Help the AI understand relationships between words
-Multiple Heads Working Together: Like having multiple readers looking at the text
-Feed-Forward Networks: Process the information further
-Transformer Blocks: Combine attention and processing
-Full Model: Puts all the pieces together to understand and generate text


Training Part (training.py)


-Reads text files to learn from
-Learns patterns in the text
-Keeps track of how well it's learning
-Saves what it learned to a file


Chat Part (chatbot.py)


-Loads the trained model
-Lets users type prompts
-Generates text responses based on what it learned

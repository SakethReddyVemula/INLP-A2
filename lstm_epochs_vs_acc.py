import os
import io
import conllu
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_tag_to_index(conllu_file):
    tag_to_index = {}
    with open(conllu_file, "r", encoding='utf-8') as f:
        data = conllu.parse(f.read())
        for sentence in data:
            for token in sentence:
                tag = token['upos']
                if tag not in tag_to_index:
                    tag_to_index[tag] = len(tag_to_index)

    return tag_to_index

def extract_columns(data, P, S):
    result = []
    for sentence in data:
        sent_tokens = []
        for token in enumerate(sentence):
            if token[1]['id'] == 1:
                for i in range(P):
                    sent_tokens.append((i + 1, "<s>", "<s>"))
                sent_tokens.append((token[1]['id'] + P, token[1]['form'], token[1]["upos"]))
            elif token[1]['id'] == len(sentence):
                sent_tokens.append((token[1]['id'] + P, token[1]['form'], token[1]["upos"]))
                for i in range(S):
                    sent_tokens.append((i + len(sentence) + P + 1, "</s>", "</s>"))
            else:
                sent_tokens.append((token[1]['id'] + P, token[1]["form"], token[1]["upos"]))
        result.append(sent_tokens)
    return result

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

def create_list_with_neighbours(lst, index, P, S):
    # Get the start and end indices for slicing
    start_index = max(0, index - P)
    end_index = min(len(lst), index + S + 1)

    # Slice the list to extract the desired elements
    new_list = lst[start_index:end_index]

    return new_list


class MyDataset(Dataset):
    def __init__(self, data, tag_to_index, vocabulary: Vocab|None=None):
        """Initialize the dataset. Setup Code goes here"""
        self.sentences = [i[0] for i in data]
        self.labels = [i[1] for i in data]
        self.tag_to_index = tag_to_index

        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN])
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            # if vocabulary provided use that
            self.vocabulary = vocabulary

    def __len__(self) -> int:
        """Returns number of datapoints"""
        return len(self.sentences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the datapoint at 'index'."""
        return torch.tensor(self.vocabulary.lookup_indices(self.sentences[index])), torch.tensor(self.labels[index])

    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of datapoints, batch them together"""
        sentences = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN]) # pad sentences with pad token id
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=torch.tensor(self.tag_to_index[PAD_TOKEN])) # pad labels with index of <pad> tag in tag_to_idx
        return padded_sentences, padded_labels

# Bidirectional = False
class LSTMTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size, embedding_dim, num_hidden_layers):
        super(LSTMTagger, self).__init__()
        self.embedding_module = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # The LSTM takes word embedding as inputs, and outputs hidden states
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_hidden_layers)

        # The linear layer that maps from hidden state space to tag sequence
        self.outputLayer = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, sentence):
        embedding = self.embedding_module(sentence)
        embedding = embedding.view(len(sentence), len(sentence[0]), -1)
        lstm_out, _ = self.lstm(embedding)
        lstm_out = lstm_out.to(device)
        output = self.outputLayer(lstm_out.view(len(sentence), len(sentence[0]), -1))
        return output
    
# Bidirectional = True
# class LSTMTagger(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, vocab_size, embedding_dim, num_hidden_layers):
#         super(LSTMTagger, self).__init__()
#         self.embedding_module = nn.Embedding(vocab_size, embedding_dim)
#         self.hidden_dim = hidden_dim
#         self.embedding_dim = embedding_dim

#         # The bidirectional LSTM takes word embedding as inputs, and outputs hidden states
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_hidden_layers, bidirectional=True)

#         # The linear layer that maps from hidden state space to tag sequence
#         self.outputLayer = nn.Linear(hidden_dim * 2, output_dim)  # *2 because of bidirectional

#     def forward(self, sentence):
#         embedding = self.embedding_module(sentence)
#         embedding = embedding.view(len(sentence), len(sentence[0]), -1)
#         lstm_out, _ = self.lstm(embedding)
#         output = self.outputLayer(lstm_out)
#         return output
    
def save_model(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    path = "./Dataset"
    train_file = os.path.join(path, "en_atis-ud-train.conllu")
    test_file = os.path.join(path, "en_atis-ud-test.conllu")
    dev_file = os.path.join(path, 'en_atis-ud-dev.conllu')

    tag_to_index_train = create_tag_to_index(train_file)
    tag_to_index_test = create_tag_to_index(test_file)
    tag_to_index_dev = create_tag_to_index(dev_file)


    tag_to_index = {**tag_to_index_train, **tag_to_index_test, **tag_to_index_dev}
    tag_to_index['<s>'] = len(tag_to_index)
    tag_to_index['</s>'] = len(tag_to_index)
    tag_to_index['<pad>'] = len(tag_to_index)

    epochs_acc = []

    for epoch_size in range(1, 16):

        # Hyper-Parameters
        EMBEDDING_DIM = 300
        BATCH_SIZE = 150
        HIDDEN_DIM = 256
        context_window = 0
        P = context_window
        S = context_window
        NUM_LAYERS = 3
        INPUT_SIZE = EMBEDDING_DIM
        OUTPUT_SIZE = len(tag_to_index)
        LEARNING_RATE = 0.001
        NUM_EPOCHS = epoch_size


        with io.open(train_file, "r", encoding='utf-8') as f:
            train_data = conllu.parse(f.read())
            train_data = extract_columns(train_data, P, S)

        with io.open(test_file, "r", encoding='utf-8') as f:
            test_data = conllu.parse(f.read())
            test_data = extract_columns(test_data, P, S)

        with io.open(dev_file, "r", encoding='utf-8') as f:
            dev_data = conllu.parse(f.read())
            dev_data = extract_columns(dev_data, P, S)

        data_train = []
        for sentence in train_data:
            words = []
            tagIndexes = []
            for token in sentence:
                words.append(token[1])
                tagIndexes.append(tag_to_index[token[2]])
            data_train.append((words, tagIndexes))

        data_test = []
        for sentence in test_data:
            words = []
            tagIndexes = []
            for token in sentence:
                words.append(token[1])
                tagIndexes.append(tag_to_index[token[2]])
            data_test.append((words, tagIndexes))

        data_dev = []
        for sentence in dev_data:
            words = []
            tagIndexes = []
            for token in sentence:
                words.append(token[1])
                tagIndexes.append(tag_to_index[token[2]])
            data_dev.append((words, tagIndexes))


        dataset_train = MyDataset(data_train, tag_to_index)
        dataset_test = MyDataset(data_test, tag_to_index, vocabulary=dataset_train.vocabulary)
        dataset_dev = MyDataset(data_dev, tag_to_index, vocabulary=dataset_train.vocabulary)
        
        model = LSTMTagger(INPUT_SIZE, HIDDEN_DIM, OUTPUT_SIZE, len(dataset_train.vocabulary), EMBEDDING_DIM, NUM_LAYERS)

        dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset_train.collate)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset_test.collate)
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset_dev.collate)


        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model = model.to(device)

        model_filename = f"lstm_model_{EMBEDDING_DIM}_{BATCH_SIZE}_{HIDDEN_DIM}_{NUM_LAYERS}_{context_window}_{LEARNING_RATE}_{NUM_EPOCHS}.pt"


        if os.path.exists(model_filename):
            model.load_state_dict(torch.load(model_filename))
        else:
            model = model.train()
            for epoch in range(NUM_EPOCHS):
                model.train()
                for batch_num, (sentences, labels) in enumerate(dataloader_train):
                    (sentences, labels) = (sentences.to(device), labels.to(device))
                    pred = model(sentences)
                    pred_flat = pred.view(-1, pred.size(2))  # [batch_size * sentence_length, output_dim]
                    labels_flat = labels.view(-1)            # [batch_size * sentence_length]

                    loss = loss_fn(pred_flat, labels_flat)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_num + 1}/{len(dataloader_train)}], Loss: {loss.item():.4f}")

            # save_model(model, model_filename)


        model.eval()
        # Inside the loop where you evaluate the model
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
        # Inside the evaluation loop
            for batch_num, (words_idx_seq, words_idx_seq_labels) in enumerate(dataloader_dev):
                (words_idx_seq, words_idx_seq_labels) = (words_idx_seq.to(device), words_idx_seq_labels.to(device))
                pred = model(words_idx_seq)

                # Flatten the predictions and labels
                pred_flat = pred.view(-1, pred.size(2))  # [batch_size * sentence_length, output_dim]
                labels_flat = words_idx_seq_labels.view(-1)  # [batch_size * sentence_length]
                _, predicted_labels = torch.max(pred_flat, 1)

                # Correct predictions count
                correct_predictions += (predicted_labels == labels_flat).sum().item()
                total_predictions += labels_flat.size(0)

        accuracy = correct_predictions / total_predictions
        print(f"Accuracy for Epochs {epoch + 1}/{NUM_EPOCHS}: {accuracy * 100:.2f}%")
        epochs_acc.append(accuracy)

    # Plotting
    plt.plot(range(0, 5), epochs_acc, marker='o')
    plt.title('context_window vs Dev set accuracy')
    plt.xlabel('context_window')
    plt.ylabel('Dev set accuracy')
    plt.grid(True)
    plt.savefig('lstm_dev_accuracy_plot.png')  # Save the plot
    plt.show()

    
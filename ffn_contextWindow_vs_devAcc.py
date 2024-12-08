import os
import io
import conllu
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

def extract_columns(data, p, s):
    result = []
    for sentence in data:
        sent_tokens = []
        for token in enumerate(sentence):
            if token[1]['id'] == 1:
                for i in range(p):
                    sent_tokens.append((i + 1, "<s>", "<s>"))
                sent_tokens.append((token[1]['id'] + p, token[1]['form'], token[1]["upos"]))
            elif token[1]['id'] == len(sentence):
                sent_tokens.append((token[1]['id'] + p, token[1]['form'], token[1]["upos"]))
                for i in range(s):
                    sent_tokens.append((i + len(sentence) + p + 1, "</s>", "</s>"))
            else:
                sent_tokens.append((token[1]['id'] + p, token[1]["form"], token[1]["upos"]))
        result.append(sent_tokens)
    return result    

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

def create_list_with_neighbours(lst, index, p, s):
    start_index = max(0, index - p)
    end_index = min(len(lst), index + s + 1)
    new_list = lst[start_index:end_index]
    return new_list


class MyDataset(Dataset):
    def __init__(self, data, p, s, vocabulary: Vocab|None=None):
        self.sentences = [i[0] for i in data]
        self.word_labels = [i[1] for i in data]

        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN])
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary
        
        self.word_index_sequences = []
        self.word_index_sequences_label = []
        for sentence, label in zip(self.sentences, self.word_labels):
            for i in range(p, len(sentence) - s):
                self.word_index_sequences.append(create_list_with_neighbours(sentence, i, p, s))
                self.word_index_sequences_label.append(label[i])
    
    def __len__(self) -> int:
        return len(self.word_index_sequences)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([self.vocabulary[word] for word in self.word_index_sequences[index]]), torch.tensor(self.word_index_sequences_label[index])
    
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size, embedding_dim, num_hidden_layers):
        super(FFNN, self).__init__()
        self.embedding_module = nn.Embedding(vocab_size, embedding_dim)
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU()
            ) for i in range(num_hidden_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, tensor_containing_indices_of_word):
        embedding = self.embedding_module(tensor_containing_indices_of_word)
        embedding = embedding.view(embedding.size(0), -1)
        
        for layer in self.hidden_layers:
            embedding = layer(embedding)
        
        output = self.output_layer(embedding)
        return output
    
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

    tag_to_index = {**tag_to_index_train, **tag_to_index_test}
    tag_to_index = {**tag_to_index, **tag_to_index_dev}
    tag_to_index['<s>'] = len(tag_to_index)
    tag_to_index['</s>'] = len(tag_to_index)
    tag_to_index['<unk>'] = len(tag_to_index)

    
    

    dev_set_accuracies = []
    
    for iter_context_window in range(0, 5):

        EMBEDDING_DIM = 300
        BATCH_SIZE = 150
        HIDDEN_DIM = 256
        NUM_LAYERS = 3
        p = iter_context_window
        s = iter_context_window
        INPUT_SIZE = (p + s + 1) * EMBEDDING_DIM
        OUTPUT_SIZE = len(tag_to_index)
        LEARNING_RATE = 0.001
        NUM_EPOCHS = 10


        with io.open(train_file, "r", encoding='utf-8') as f:
            train_data = conllu.parse(f.read())
            train_data = extract_columns(train_data, p, s)

        with io.open(test_file, "r", encoding='utf-8') as f:
            test_data = conllu.parse(f.read())
            test_data = extract_columns(test_data, p, s)

        with io.open(dev_file, "r", encoding='utf-8') as f:
            dev_data = conllu.parse(f.read())
            dev_data = extract_columns(dev_data, p, s)

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

        p = iter_context_window
        s = iter_context_window

        dataset_train = MyDataset(data_train, p, s)   
        dataset_test = MyDataset(data_test, p, s, vocabulary=dataset_train.vocabulary)
        dataset_dev = MyDataset(data_dev, p, s, vocabulary=dataset_train.vocabulary)

        dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=BATCH_SIZE, shuffle=False)

        model = FFNN(INPUT_SIZE, HIDDEN_DIM, OUTPUT_SIZE, len(dataset_train.vocabulary), EMBEDDING_DIM, NUM_LAYERS)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        model = model.to(device)

        model_filename = f"ffn_model_{EMBEDDING_DIM}_{BATCH_SIZE}_{HIDDEN_DIM}_{NUM_LAYERS}_{iter_context_window}_{LEARNING_RATE}_{NUM_EPOCHS}.pt"


        if os.path.exists(model_filename) and False:
            model.load_state_dict(torch.load(model_filename))
        else:
            model = model.train()
            for epoch in range(NUM_EPOCHS):
                model.train()
                for batch_num, (words_idx_seq, words_idx_seq_labels) in enumerate(dataloader_train):
                    (words_idx_seq, words_idx_seq_labels) = (words_idx_seq.to(device), words_idx_seq_labels.to(device))
                    pred = model(words_idx_seq)
                    loss = loss_fn(pred, words_idx_seq_labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_num + 1}/{len(dataloader_train)}], Loss: {loss.item():.4f}")

            # save_model(model, model_filename)

        model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_num, (words_idx_seq, words_idx_seq_labels) in enumerate(dataloader_dev):
                (words_idx_seq, words_idx_seq_labels) = (words_idx_seq.to(device), words_idx_seq_labels.to(device))
                pred = model(words_idx_seq)
                _, predicted_labels = torch.max(pred, 1)
                correct_predictions += (predicted_labels == words_idx_seq_labels).sum().item()
                total_predictions += words_idx_seq_labels.size(0)
        
        accuracy = correct_predictions / total_predictions
        print(f"Context Window: {iter_context_window} Accuracy: {accuracy * 100:.2f}%")
        dev_set_accuracies.append(accuracy * 100)

    # Plotting
    plt.plot(range(0, 5), dev_set_accuracies, marker='o')
    plt.title('context_window vs Dev set accuracy')
    plt.xlabel('context_window')
    plt.ylabel('Dev set accuracy')
    plt.grid(True)
    plt.savefig('ffn_dev_accuracy_plot.png')  # Save the plot
    plt.show()

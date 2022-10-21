import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

# batch_x = []
# batch_x.extend(["x"]*3)
# batch_x.extend(["y"]*3)
# print(batch_x)

class Dataset:
    """
    输入语料，形式如["句子1" "句子2" "句子3"]
    使用下标直接返回某一数据 无minibatch，即batchsize=训练数据个数
    """

    def __init__(self, corpus, context_size=2):
        self.idx2word = list()
        self.word2idx = dict()
        self.train_data_targets = []
        self.train_data_word = []
        all_words = set()
        for sentence in corpus:
            all_words.update([word for word in sentence.split()])
        self.idx2word = list(all_words)
        for i, word in enumerate(self.idx2word):
            self.word2idx[word] = i
        for sentence in tqdm(corpus, desc="processing data"):
            sentence = sentence.split()
            target = []
            if len(sentence) < (2 * context_size + 1):
                continue
            sentence = [self.word2idx[word] for word in sentence]
            for i in range(context_size, len(sentence) - context_size):
                for j in range(i - context_size,i):     #我采用的是预测一个词，一个中心词对应多个周边词
                    target = sentence[j]
                    word= sentence[i]
                    self.train_data_targets.append(target)
                    self.train_data_word.append(word)
                for j in range(i + 1, i + context_size + 1):
                    target = sentence[j]
                    word = sentence[i]
                    self.train_data_targets.append(target)
                    self.train_data_word.append(word)
                # target = sentence[i - context_size:i] + sentence[i + 1:i + context_size + 1]   #在skip-gram中context是需要预测的目标
                # word= sentence[i]        #中心词
                # # target.append([word]*len(target_x))  #中心词扩张，因为一个中心词对应多个周边词
                # self.train_data_targets.append(target)
                # self.train_data_word.append(word)

    def len_of_vocab(self):
        return len(self.idx2word)

    def getdata(self):
        targets = torch.tensor(self.train_data_targets)
        inputs = torch.tensor(self.train_data_word)
        print(len(targets))
        return inputs, targets


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)



    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = embeds.mean(dim=1)
        output = self.output(embeds)
        # output = self.output(hidden)      #skip-gram中没有融合层
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


embedding_dim = 8       #词向量维度
context_size = 2        #窗口大小
num_epoch = 2000

corpus = ["w a s d q w",
          "q w e r m n l",
          "w o r s v n y f s d d t u",
          "q b b f z p m r x g f a a"]

dataset = Dataset(corpus, context_size)
nll_loss = nn.NLLLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkipGramModel(dataset.len_of_vocab(), embedding_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in tqdm(range(num_epoch), desc="Training"):
    inputs, targets = [x.to(device) for x in dataset.getdata()]
    optimizer.zero_grad()
    log_probs = model(inputs)
    loss = nll_loss(log_probs, targets)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.2f}")


def save_pretrained(vocab, embeds, save_path):
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx2word):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
    print(f"Pretrained embeddings saved to: {save_path}")


save_pretrained(dataset, model.embeddings.weight.data, "cbow.simple.vec")
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') #open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer

        if len(word) <= n_step:   #pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index+n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch

def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  #open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))   #set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    #add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"

    return word2number_dict, number2word_dict

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)

        '''define the parameter of RNN'''
        '''begin'''
        ##Complete this code
        self.W_ax = nn.Linear(emb_size,n_hidden,bias=False)
        self.W_aa = nn.Linear(n_hidden,n_hidden,bias=False)
        self.b_a = nn.Parameter(torch.ones([n_hidden]))
        '''end'''

        self.tanh = nn.Tanh()

        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1) # X : [n_step, batch_size, emb_size]       第0维和第1维转置
        sample_size = X.size()[1]
        # print(f"sample_size={sample_size}")
        '''do this RNN forward'''
        '''begin'''
        ##Complete your code with the hint: a^(t) = tanh(W_{ax}x^(t)+W_{aa}a^(t-1)+b_{a})  y^(t)=softmx(Wa^(t)+b)
        # a_0 = torch.zeros(1,n_hidden)
        # a = a_0
        a = torch.zeros(sample_size,n_hidden)
        # output = torch.zeros(n_step,batch_size,n_hidden)
        # print(f"a.shape={a.shape}")
        # print(f"n_step={n_step}\nbatch_size={batch_size}\nn_class={n_class}\nemb_size:{emb_size}")
        for x in X:
            # print(f"x.shape={x.shape}")
            a = self.tanh(self.W_ax(x)+self.W_aa(a)+self.b_a)
        '''end'''
        model_output = torch.tanh(self.W(a)+self.b)
        return model_output

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)

        '''define the parameter of LSTM'''
        '''begin'''
        ##Complete this code
        #输入门参数
        #第一层LSTM
        self.U_i1 = nn.Linear(emb_size,n_hidden,bias=False)
        self.V_i1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_i1 = nn.Parameter(torch.ones([n_hidden]))
        #第二层LSTM
        self.U_i2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.V_i2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_i2 = nn.Parameter(torch.ones([n_hidden]))

        #遗忘门参数
        #第一层LSTM
        self.U_f1 = nn.Linear(emb_size, n_hidden, bias=False)
        self.V_f1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_f1 = nn.Parameter(torch.ones([n_hidden]))
        #第二层LSTM
        self.U_f2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.V_f2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_f2 = nn.Parameter(torch.ones([n_hidden]))


        #记忆参数
        #第一层LSTM
        self.U_c1 = nn.Linear(emb_size, n_hidden, bias=False)
        self.V_c1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_c1 = nn.Parameter(torch.ones([n_hidden]))
        #第二层LSTM
        self.U_c2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.V_c2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_c2 = nn.Parameter(torch.ones([n_hidden]))

        #输入门参数
        #第一层LSTM
        self.U_o1 = nn.Linear(emb_size, n_hidden, bias=False)
        self.V_o1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_o1 = nn.Parameter(torch.ones([n_hidden]))
        #第二层LSTM
        self.U_o2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.V_o2 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_o2 = nn.Parameter(torch.ones([n_hidden]))

        '''end'''

        self.tanh = nn.Tanh()

        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1) # X : [n_step, batch_size, emb_size]       第0维和第1维转置
        sample_size = X.size()[1]
        # print(f"sample_size={sample_size}")
        '''do this LSTM forward'''
        '''begin'''
        ##Complete your code with the hint: a^(t) = tanh(W_{ax}x^(t)+W_{aa}a^(t-1)+b_{a})  y^(t)=softmx(Wa^(t)+b)
        h_t1 = torch.zeros(sample_size,n_hidden)
        c_t1 = torch.zeros(sample_size,n_hidden)

        h_t2 = torch.zeros(sample_size, n_hidden)
        c_t2 = torch.zeros(sample_size, n_hidden)

        for x in X:
            # print(f"x.shape={x.shape}")
            # a = self.tanh(self.W_ax(x)+self.W_aa(a)+self.b_a)
            # print(torch.cat([h_t,x],1).shape)
            # print(h_t.shape,x.shape)
            #第一层LSTM
            f_t1 = torch.sigmoid(self.U_f1(x)+self.V_f1(h_t1)+self.b_f1)
            i_t1 = torch.sigmoid(self.U_i1(x)+self.V_i1(h_t1)+self.b_i1)
            g_t1 = self.tanh(self.U_c1(x)+self.V_c1(h_t1)+self.b_c1)
            #print(f"ft.shape={f_t.shape}\nct.shape={c_t.shape}\ni_t.shape={i_t.shape}\ngt.shape={g_t.shape}")
            c_t1 = torch.mul(c_t1,f_t1)+torch.mul(i_t1,g_t1)
            o_t1 = torch.sigmoid(self.U_o1(x)+self.V_o1(h_t1)+self.b_o1)
            h_t1 = torch.mul(o_t1,self.tanh(c_t1))



            #第二层LSTM
            f_t2= torch.sigmoid(self.U_f2(h_t1) + self.V_f2(h_t2) + self.b_f1)
            i_t2 = torch.sigmoid(self.U_i2(h_t1) + self.V_i2(h_t2) + self.b_i1)
            g_t2 = self.tanh(self.U_c2(h_t1) + self.V_c2(h_t2) + self.b_c1)
            c_t2 = torch.mul(c_t2, f_t2) + torch.mul(i_t2, g_t2)
            o_t2 = torch.sigmoid(self.U_o2(h_t1) + self.V_o2(h_t2) + self.b_o1)
            h_t2 = torch.mul(o_t2, self.tanh(c_t2))



        model_output = self.W(h_t2) + self.b
        '''end'''

        return model_output

def train_lstmlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 50 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'lost =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(word2number_dict, n_step)
        all_valid_batch.to(device)
        all_valid_target.to(device)
        
        total_valid = len(all_valid_target)*128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1
          
            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'lost =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/lstmlm_model_epoch{epoch+1}.ckpt')

def test_lstmlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  #load the selected model

    #load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(word2number_dict, n_step)
    total_test = len(all_test_target)*128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('lost =','{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 5 # number of hidden units in one cell
    batch_size = 512 #batch size
    learn_rate = 0.001
    all_epoch = 200 #the all epoch for training
    emb_size = 128 #embeding size
    save_checkpoint_epoch = 100 # save a checkpoint per save_checkpoint_epoch epochs
    train_path = 'data/train.txt' # the path of train dataset

    word2number_dict, number2word_dict = make_dict(train_path) #use the make_dict function to make the dict
    print("The size of the dictionary is:", len(word2number_dict))

    n_class = len(word2number_dict)  #n_class (= dict size)

    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    print("The number of the train batch is:", len(all_input_batch))

    all_input_batch = torch.LongTensor(all_input_batch).to(device)   #list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)

    # print("\nTrain the LSTMLM……………………")
    # train_lstmlm()

    print("\nTest the LSTMLM……………………")
    # select_model_path = "models/lstmlm_model_epoch100.ckpt"
    select_model_path = "models/lstmlm_model_epoch200.ckpt"
    test_lstmlm(select_model_path)
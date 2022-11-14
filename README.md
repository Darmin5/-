# NLP works

## 2022年秋季学期自然语言处理作业

### 作业1：skip-gram模型实现
完整代码在hw1_skip-gram.py文件中

 * 主要修改代码：<br>
```python
# 1.在数据处理阶段
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
```

```python
#2.在前向传播阶段
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = embeds.mean(dim=1)
        output = self.output(embeds)
        # output = self.output(hidden)      #skip-gram中没有融合层
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
```

### 作业2：RNN神经网络
项目文件都在rnnlm文件夹中

* 主要补充代码：<br>
```python
#1.模型参数
'''define the parameter of RNN'''
        '''begin'''
        ##Complete this code
        self.W_ax = nn.Linear(emb_size,n_hidden,bias=False)
        self.W_aa = nn.Linear(n_hidden,n_hidden,bias=False)
        self.b_a = nn.Parameter(torch.ones([n_hidden]))
        '''end'''

```

```python
#2.前向传播阶段
'''do this RNN forward'''
        '''begin'''
        a = torch.zeros(sample_size,n_hidden)
        for x in X:
            a = self.tanh(self.W_ax(x)+self.W_aa(a)+self.b_a)
        model_output = self.W(a)+self.b
        '''end'''
```

### 课程项目：双层LSTM
项目文件在NLP_LSTM_assignment文件夹中

* 主要补充代码（这里只列出单层网络的，第二层与第一层类似，项目文件中实现的是双层LSTM）
```python
#1.网络的参数设置
        '''define the parameter of LSTM'''
        '''begin'''
##Complete this code
        #输入门参数
        self.U_i1 = nn.Linear(emb_size,n_hidden,bias=False)
        self.V_i1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_i1 = nn.Parameter(torch.ones([n_hidden]))
#遗忘门参数
        self.U_f1 = nn.Linear(emb_size, n_hidden, bias=False)
        self.V_f1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_f1 = nn.Parameter(torch.ones([n_hidden]))
#记忆参数
        self.U_c1 = nn.Linear(emb_size, n_hidden, bias=False)
        self.V_c1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_c1 = nn.Parameter(torch.ones([n_hidden]))
 #输入门参数
        self.U_o1 = nn.Linear(emb_size, n_hidden, bias=False)
        self.V_o1 = nn.Linear(n_hidden, n_hidden, bias=False)
        self.b_o1 = nn.Parameter(torch.ones([n_hidden]))
        '''end'''
```
```python
#2.前向传播
'''do this LSTM forward'''
        '''begin'''
        ##Complete your code with the hint: a^(t) = tanh(W_{ax}x^(t)+W_{aa}a^(t-1)+b_{a})  y^(t)=softmx(Wa^(t)+b)
        h_t1 = torch.zeros(sample_size,n_hidden)
        c_t1 = torch.zeros(sample_size,n_hidden)
        
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
        '''end'''

```

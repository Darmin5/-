# NLP works

## 2022年秋季学期自然语言处理作业


  * 作业1：skip-gram模型实现

    * 主要修改代码：
    * 在数据处理阶段
```python
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
    * 在前向传播阶段
```python
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = embeds.mean(dim=1)
        output = self.output(embeds)
        # output = self.output(hidden)      #skip-gram中没有融合层
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
```

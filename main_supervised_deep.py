import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchnlp.encoders.text.text_encoder import pad_tensor

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score

import dataclass
import models

parser = argparse.ArgumentParser(description='spam detection using supervised deep learning')
parser.add_argument(
    '--dataset',
    default='Heyspam',
    help='Heyspam')
parser.add_argument(
    '--model', default='textcnn', help='textcnn')
parser.add_argument('--embedding_size', type=int, default=100)
parser.add_argument('--sequence_length', type=int, default=512)
parser.add_argument('--filter_sizes', default=[3, 4, 5])
parser.add_argument('--num_filters', type=int, default=512)
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train')
parser.add_argument(
    '--lr', type=float, default=0.001, help='learning rate')


args = parser.parse_args()

dataset = getattr(dataclass, args.dataset)(is_deep=True, is_jieba=True, is_balanced=True)


train_padded = [pad_tensor(tensor, args.sequence_length, dataset.stoi['[PAD]']) for tensor in dataset.text_train]
train_padded = torch.stack(train_padded, dim=0).contiguous()
train_labels = torch.stack(dataset.label_train)


if args.model == "textcnn":
    model = models.TextCNN(args, vocab_size=len(dataset.stoi), num_classes=dataset.n_classes)   

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training
for epoch in range(args.epochs):
    optimizer.zero_grad()
    output = model(train_padded)

    # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, train_labels)
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Test
test_padded = [pad_tensor(tensor, args.sequence_length, dataset.stoi['[PAD]']) for tensor in dataset.text_test]
test_padded = torch.stack(test_padded, dim=0).contiguous()

# Predict
pre = model(test_padded).data.max(1)[1]
print(pre.shape)
 
prec, recall, f1, _ = precision_recall_fscore_support(dataset.label_test.data, pre, average="weighted")  
acc = accuracy_score(dataset.label_test.data, pre)
auc = roc_auc_score(dataset.label_test.data, pre)
print("prec:{} ; reacll:{} ; f1:{} ; acc:{} ; auc:{}".format(prec, recall, f1, acc, auc))

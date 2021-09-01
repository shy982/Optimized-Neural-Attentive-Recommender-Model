import time
import csv
import pickle
import numpy as np
import pandas as pd
import operator

df = pd.read_csv(r"/content/drive/My Drive/NARM/train-item-views.csv",sep = ';')
df['prices'] = 1000 #please run the script with whichever mean you want here
df['prices'] = df.prices.apply(np.random.normal)
df.to_csv(r"/content/drive/My Drive/NARM/train-item-views1.csv",sep = ";")
# Load .csv dataset
with open(r"/content/drive/My Drive/NARM/train-item-views1.csv", "r") as f:
    reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['sessionId']
        if curdate and not curid == sessid:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        item = data['itemId']
        curdate = data['eventdate']
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
        if ctr % 100000 == 0:
            print ('Loaded', ctr)
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    sess_date[curid] = date

# Filter out length 1 sessions
for s in list(sess_clicks.keys()):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
   seq = sess_clicks[s]
    for iid in seq:
      if iid in iid_counts:
        iid_counts[iid] += 1
        else:
           iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

for s in list(sess_clicks.keys()):
    curseq = sess_clicks[s]
    filseq = filter(lambda i: iid_counts[i] >= 5, curseq)
    length = sum(1 for _ in filseq)
    if length < 2:
        del sess_clicks[s]
        del sess_date[s]
    else:
       sess_clicks[s] = filseq

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = maxdate - 86400 * 7
print('Split date', splitdate)
train_sess = filter(lambda x: x[1] < splitdate, dates)
test_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
train_sess = sorted(train_sess, key=operator.itemgetter(1))
test_sess = sorted(test_sess, key=operator.itemgetter(1))

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
item_ctr = 1
train_seqs = []
train_dates = []
print("Running Train")
# Convert training sessions to sequences and renumber items to start from 1
for s, date in train_sess:
    seq = sess_clicks[s]
    outseq = []
    for i in list(seq):
        if i in list(item_dict.keys()):
            outseq += [item_dict[i]]
        else:
            outseq += [item_ctr]
            item_dict[i] = item_ctr
            item_ctr += 1
        print(item_ctr)
    if len(outseq) < 2:  # Doesn't occur
        continue
    train_seqs += [outseq]
    train_dates += [date]
    if item_ctr > 719470: 
            break
print("Running Test")
test_seqs = []
test_dates = []
h = 0
# Convert test sessions to sequences, ignoring items that do not appear in training set
for s, date in test_sess:
    seq = sess_clicks[s]
    outseq = []
    for i in seq:
        if i in item_dict.keys():
            outseq += [item_dict[i]]
    if len(outseq) < 2:
        continue
    test_seqs += [outseq]
    test_dates += [date]
    h = h+1
    if h>60858:
        break
print(item_ctr)

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    for seq, date in zip(iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]

    return out_seqs, out_dates, labs


tr_seqs, tr_dates, tr_labs = process_seqs(train_seqs,train_dates)
te_seqs, te_dates, te_labs = process_seqs(test_seqs,test_dates)

train = (tr_seqs, tr_labs)
test = (te_seqs, te_labs)

f1 = open('/content/drive/My Drive/NARM/digi-train.pkl', 'wb')
pickle.dump(train, f1)
f1.close()
f2 = open('/content/drive/My Drive/NARM/digi-test.pkl', 'wb')
pickle.dump(test, f2)
f2.close()

print('Done.')

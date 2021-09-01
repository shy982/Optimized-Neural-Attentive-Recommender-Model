import tensorflow as tf
import numpy as np
from data_process_tf import load_data,prepare_data
import time
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn

dim_proj = 50  # word embeding dimension
hidden_units = 100  # GRU number of hidden units.
patience = 100  # Number of epoch to wait before early stop if no progress
max_epochs = 30  # The maximum number of epoch to run
dispFreq = 100  # Display to stdout the training progress every N updates
lrate = 0.001 # Learning rate
n_items = 55001  # Vocabulary size
encoder = 'gru',  # TODO: can be removed must be gru.
saveto = 'gru_model.npz',  # The best model will be saved there
is_valid = True  # Compute the validation error after this number of update.
is_save = False  # Save the parameters after every saveFreq updates
batch_size = 512  # The batch size during training.
valid_batch_size = 512  # The batch size used for validation/test set.
dataset = 'rsc2015'

tf.reset_default_graph()
sess= tf.Session()
#Inputs
input_items = tf.placeholder('int32',[None,None],name='items')
input_item_mask = tf.placeholder('bool',[None,None],name='item_mask')
target_items = tf.placeholder('int32',[None],name='target_items')
keep_prob_2 = tf.placeholder(tf.float32, name='keep_prob_2')
keep_prob_1 = tf.placeholder(tf.float32, name='keep_prob_1')


init_weights = tf.random_normal_initializer(-0.5, 0.5)
input_batch_size = tf.shape(input_items)[0]

with tf.variable_scope("item_emb"), tf.device("/cpu:0"):
    item_emb_mat = tf.get_variable("item_emb_mat", dtype='float', shape=[n_items, dim_proj], initializer=init_weights)
    with tf.name_scope("word"):
        Ax = tf.nn.embedding_lookup(item_emb_mat, input_items) #[batch_size,session length,dim_proj]

d_cell_fw = GRUCell(hidden_units)
cell_fw = tf.contrib.rnn.DropoutWrapper(d_cell_fw, output_keep_prob=keep_prob_2,input_keep_prob=keep_prob_1)
input_len = tf.reduce_sum(tf.cast(input_item_mask, 'int32'), 1)  # [batch_size]
with tf.variable_scope("encoder"):
    #https://stackoverflow.com/questions/48238113/tensorflow-dynamic-rnn-state/48239320#48239320
    outputs, final_state = dynamic_rnn(cell_fw, Ax,dtype=tf.float32,scope='encoder_rnn',sequence_length=input_len)#[batch_size,max_enc_steps,hidden_units]

    with tf.variable_scope("attention"):
        A1 = tf.get_variable(name='A1',shape=[hidden_units,hidden_units],dtype=tf.float32,initializer=init_weights)
        A2 = tf.get_variable(name='A2',shape=[hidden_units,hidden_units],dtype=tf.float32,initializer=init_weights)
        A3 = tf.get_variable(name='A3',shape=[hidden_units,hidden_units],dtype=tf.float32,initializer=init_weights)
        
        v_blend = tf.get_variable(name="v_blend", shape=[1, hidden_units], dtype=tf.float32, initializer=init_weights)

        #q(ht ,hj ) = vTσ(A1ht + A2hj ) similarity score
        final_state_portion = tf.expand_dims(tf.matmul(final_state,A2),1) #[batch_size,1,hidden_units]
        individual_state_portion = tf.reshape(tf.matmul(tf.reshape(outputs,[-1,hidden_units]),A1),[input_batch_size,-1,hidden_units])#[batch_size,max_enc_steps,hidden_units]
        #in the paper they do not normalize this using softmax ..?
        attention_weights = tf.reduce_sum(v_blend * tf.nn.sigmoid(individual_state_portion + final_state_portion), 2) #(batch_size, max_enc_steps)
        context_vector_local = tf.reduce_sum(tf.reshape(attention_weights,[input_batch_size,-1,1]) * outputs ,1) #(batch_size,hidden_units)
        final_context_vector = tf.concat([final_state,context_vector_local],axis=1) #(batch_size,2*hidden_units)

with tf.variable_scope("decoder"):
    B = tf.get_variable(name="B",shape=[dim_proj,2*hidden_units],dtype=tf.float32,initializer=init_weights)
    decoded_outputs = tf.transpose(tf.matmul(item_emb_mat,tf.matmul(B,tf.transpose(final_context_vector))))#num_items*batch_size
    output_probs = tf.nn.softmax(decoded_outputs)

with tf.name_scope("optimization"):
    # Loss function
    ce_loss  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_items,logits=decoded_outputs))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(ce_loss)

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

def numpy_floatX(data):
    return np.asarray(data, dtype=float)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def pred_evaluation(prepare_data, data, iterator):
    """
    Compute recall@20 and mrr@20
    prepare_data: usual prepare_data for that dataset.
    """
    recall = 0.0
    mrr = 0.0
    evalutation_point_count = 0
    # pred_res = []
    # att = []

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index])
        preds = sess.run(output_probs,feed_dict={input_items: x,target_items : y,input_item_mask :mask,keep_prob_1:1.0,keep_prob_2:1.0})
        # weights = f_weight(x, mask)
        targets = y
        ranks = (preds.T > np.diag(preds.T[targets])).sum(axis=0) + 1
        rank_ok = (ranks <= 20)

        # pred_res += list(rank_ok)
        recall += rank_ok.sum()
        mrr += (1.0 / ranks[rank_ok]).sum()
        evalutation_point_count += len(ranks)
        # att.append(weights)
    recall = numpy_floatX(recall) / evalutation_point_count
    mrr = numpy_floatX(mrr) / evalutation_point_count
    eval_score = (recall, mrr)
    return eval_score

print('Loading data')
train, valid, test = load_data()

print("%d train examples" % len(train[0])) #train[0] = x, train[1] =y
print("%d valid examples" % len(valid[0]))
print("%d test examples" % len(test[0]))

history_errs = []
history_vali = []
bad_count = 0

uidx = 0  # the number of update done
estop = False  # early stop

try:
    for eidx in range(max_epochs):
        start_time = time.time()
        n_samples = 0
        epoch_loss = []

        # Get new shuffled index for the training set.
        kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
        kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
        kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

        for _, train_index in kf:
            uidx += 1

            # Select the random examples for this minibatch
            y = [train[1][t] for t in train_index]
            x = [train[0][t]for t in train_index]

            # Get the data in numpy.ndarray format
            # This swap the axis!
            # Return something of shape (minibatch maxlen, n samples)
            x, mask, y = prepare_data(x, y)
            n_samples += x.shape[0]

            _,loss = sess.run([optimizer,ce_loss],feed_dict={input_items: x,target_items : y,input_item_mask :mask,keep_prob_1:0.75,keep_prob_2:0.5})
            epoch_loss.append(loss)
            if np.isnan(loss) or np.isinf(loss):
                print('bad loss detected: ', loss)

            if np.mod(uidx, dispFreq) == 0:
                print('Epoch ', eidx, 'Update ', uidx, 'Loss ', np.mean(epoch_loss))

        if is_valid:
            valid_evaluation = pred_evaluation(prepare_data, valid, kf_valid)
            test_evaluation = pred_evaluation(prepare_data, test, kf_test)
            history_errs.append([valid_evaluation, test_evaluation])

            if len(history_vali) == 0 or valid_evaluation[0] >= np.array(history_vali).max():

                save_path = saver.save(sess, "/tmp/model.ckpt")
                print('Best perfomance updated!')
                bad_count = 0

            print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1],
                  '\nTest Recall@20', test_evaluation[0], '   Test Mrr@20:', test_evaluation[1])

            if len(history_vali) > 10 and valid_evaluation[0] <= np.array(history_vali).max():
                bad_count += 1
                print('===========================>Bad counter: ' + str(bad_count))
                print('current validation recall: ' + str(valid_evaluation[0]) +
                      '      history max recall:' + str(np.array(history_vali).max()))
                if bad_count > patience:
                    print('Early Stop!')
                    estop = True

            history_vali.append(valid_evaluation[0])

        end_time = time.time()
        print('Seen %d samples' % n_samples)
        #print(('This epoch took %.1fs' % (end_time - start_time)), file=sys.stderr)
        print(('This epoch took %.1fs' % (end_time - start_time)))

        if estop:
            break

except KeyboardInterrupt:
    print("Training interupted")

saver.restore(sess, "/tmp/model.ckpt")
kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
valid_evaluation = pred_evaluation(prepare_data, valid, kf_valid)
test_evaluation = pred_evaluation(prepare_data, test, kf_test)

print('=================Best performance=================')
print('Valid Recall@20:', valid_evaluation[0], '   Valid Mrr@20:', valid_evaluation[1],
      '\nTest Recall@20', test_evaluation[0], '   Test Mrr@20:', test_evaluation[1])
print('==================================================')

#input: number of records * length of sequence * types of sequences
#number of records = number of students for training
#length of sequence = summation of max_code_states for each question
#types of sequences = 1


#using truncated back propagation through time(TBPTT)

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rnn_model(max_code_states, train_input_vectors, train_input_labels, test_input_vectors, test_input_labels, num_code_states, learning_rate, nepoch, hidden_dim, output_dim = 1, bptt_truncate = 5, min_clip_value = -10, max_clip_value = 10):
    
    weight_input_hidden = np.random.uniform(0, 1, (hidden_dim, num_code_states))    #ih
    weight_hidden = np.random.uniform(0, 1, (hidden_dim, hidden_dim))   #hh
    weight_hidden_output = np.random.uniform(0, 1, (output_dim, hidden_dim))    #ho

    train_data_param = []
    train_data_label = []
    test_data_param = []
    test_data_label = []

    #for all training instances
    for i in range(len(train_input_vectors)):   #each student
        for j in range(len(train_input_vectors[i])):    #each vector
            train_data_param[i].append(train_input_vectors[i][:j+1])   #storing inputs: list of lists
            train_data_label[i].append(train_input_labels[i][j])     #storing labels: list of lists
    
    for i in range(len(test_input_vectors)):
        for j in range(len(test_input_vectors[i])):
            test_data_param[i].append(test_input_vectors[i][:j+1])
            test_data_label[i].append(test_input_labels[i][j])

    train_data_param = np.array(train_data_param)
    train_data_param = np.expand_dims(train_data_param, axis = 3)

    test_data_param = np.array(test_data_param)
    test_data_param = np.expand_dims(test_data_param, axis = 3)

    train_data_label = np.array(train_data_label)
    train_data_label = np.expand_dims(train_data_label, axis = 2)

    test_data_label = np.array(test_data_label)
    test_data_label = np.expand_dims(test_data_label, axis = 2)

    for i in range(len(train_data_label)):
        loss = 0.0
        for j in range(len(train_data_label[i])):
            input, label = train_data_param[i][j], train_data_label[i][j]
            prev_state = np.zeros((hidden_dim, 1))
            for t in range(len(train_data_label[i])):
                new_input = np.zeros(input.shape)
                new_input[t] = input[t]
                mul_ih = np.dot(weight_input_hidden, new_input)
                mul_hh = np.dot(weight_hidden, prev_state)
                add = mul_ih + mul_hh
                state = sigmoid(add)
                mul_ho = np.dot(weight_hidden_output, state)
                prev_state = state
            loss_per_record = (label - mul_ho)**2 / 2
            loss += loss_per_record
        loss = loss / float(label.shape[0])

        test_loss = 0.0
        for j in range(len(test_data_label[i])):
            input, label = test_data_param[i][j], test_data_label[i][j]
            prev_state = np.zeros((hidden_dim, 1))
            for t in range(len(test_data_label[i])):
                new_input = np.zeros(input.shape)
                new_input[t] = input[t]
                mul_ih = np.dot(weight_input_hidden, new_input)
                mul_hh = np.dot(weight_hidden, prev_state)
                add = mul_ih + mul_hh
                state = sigmoid(add)
                mul_ho = np.dot(weight_hidden_output, state)
                prev_state = state
            loss_per_record = (label - mul_ho)**2 / 2
            test_loss += loss_per_record
        test_loss = test_loss / float(label.shape[0])

        print('student: ', i + 1, ', Loss: ', loss, ', test_loss: ', test_loss)

        for j in range(len(train_data_label[i])):
            input, label = train_data_param[i][j], train_data_label[i][j]
            prev_state = np.zeros((hidden_dim, 1))
            layers = []
            train_ih = np.zeros(weight_input_hidden.shape)
            train_ho = np.zeros(weight_hidden_output.shape)
            train_hh = np.zeros(weight_hidden.shape)
            train_ih_t = np.zeros(weight_input_hidden.shape)
            train_ho_t = np.zeros(weight_hidden_output.shape)
            train_hh_t = np.zeros(weight_hidden.shape)
            train_ih_i = np.zeros(weight_input_hidden.shape)
            train_hh_i = np.zeros(weight_hidden.shape)

        #forward pass
            for t in range(len(train_data_label[i])):
                new_input = np.zeros(input.shape)
                new_input[t] = input[t]
                mul_ih = np.dot(weight_input_hidden, new_input)
                mul_hh = np.dot(weight_hidden, prev_state)
                add = mul_ih + mul_hh
                state = sigmoid(add)
                mul_ho = np.dot(weight_hidden_output, state)
                layers.append({'state': state, 'prev_state': prev_state})
                prev_state = state
        
            dmul_ho = mul_ho - label

            #back propagation
            for t in range(len(train_data_label[i])):
                train_ho_t = np.dot(dmul_ho, np.transpose(layers[t]['state']))
                ds_train_ho = np.dot(np.transpose(weight_hidden_output), dmul_ho)
                ds = ds_train_ho
                dadd = add * (1 - add)*ds
                dmul_hh = dadd * np.ones_like(mul_hh)
                dprev_state = np.dot(np.transpose(weight_hidden), dmul_hh)

                for k in range(t-1, max(-1, t-bptt_truncate-1), -1):
                    ds = ds_train_ho + dprev_state
                    dadd = add * (1-add)*ds
                    dmul_hh = dadd * np.ones_like(mul_hh)
                    dmul_ih = dadd * np.ones_like(mul_ih)
                    train_hh_i = np.dot(weight_hidden, layers[t]['prev_state'])
                    dprev_state = np.dot(np.transpose(weight_hidden), dmul_hh)
                    
                    new_input = np.zeros(input.shape)
                    new_input[t] = input[t]
                    train_ih_i = np.dot(weight_input_hidden, new_input)
                    dinput = np.dot(np.transpose(weight_input_hidden), dmul_ih)
                    train_ih_t += train_ih_i
                    train_hh_t += train_hh_i
                train_ho += train_ho_t
                train_ih += train_ih_t
                train_hh += train_hh_t
                if train_ih.max() > max_clip_value:
                    train_ih[train_ih > max_clip_value] = max_clip_value
                if train_ho.max() > max_clip_value:
                    train_ho[train_ho > max_clip_value] = max_clip_value
                if train_hh.max() > max_clip_value:
                    train_hh[train_hh > max_clip_value] = max_clip_value
                
                if train_ih.min() < min_clip_value:
                    train_ih[train_ih < min_clip_value] = min_clip_value
                if train_ho.min() < min_clip_value:
                    train_ho[train_ho < min_clip_value] = min_clip_value
                if train_hh.min() < min_clip_value:
                    train_hh[train_hh < min_clip_value] = min_clip_value
            weight_input_hidden -= learning_rate * train_ih
            weight_hidden_output -= learning_rate * train_ho
            weight_hidden -= learning_rate * train_hh
    
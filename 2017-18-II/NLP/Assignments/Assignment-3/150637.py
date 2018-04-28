
# coding: utf-8

# In[29]:


from conllu import parse, parse_tree
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import re
import gensim
import networkx as nx
from sklearn.neural_network import MLPClassifier


# In[27]:


def load_data_file(file_name):
    file = open(file_name, "r")
    text = (file.read()).decode('utf-8')
    return parse(text)



# In[4]:


glove_size = 50

num_relations = 37

max_token_id = 1000

model_size = (150, 100)

tags = dict([(el, i) for i, el in enumerate(
    ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', 'CCONJ',
      
     'DET', 'NUM', 'PART', 'PRON', 'SCONJ', 'PUNCT', 'SYM', 'X']
)])

tags_len = len(tags.keys())

relations = dict([(el, i) for i, el in enumerate(
    ['nsubj', 'obj', 'iobj', 'obl', 'vocative', 'expl', 'dislocated', 'nmod', 'appos', 'nummod', 'conj', 'cc', 'fixed', 'flat', 'compound', 'csubj', 'ccomp', 'xcomp', 'advcl',
        'acl', 'root', 'advmod', 'discourse', 'amod', 'aux', 'cop', 'mark', 'det', 'clf', 'case', 'dep', 'list', 'parataxis', 'orphan', 'goeswith', 'reparandum', 'punct']
)])

relations_len = len(relations.keys())


# In[5]:


tags_len


# In[6]:


vocab = []
file = "/home/saransh/Desktop/nlp assignment 2/glove.6B.50d.txt"
def loadGloveModel(gloveFile):
    print ("Loading Glove Model") 
    with open(gloveFile) as f:
        content = f.readlines()
    
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        vocab.append(word)
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
     
glove_vectors = loadGloveModel(file)


# In[21]:


def get_features(id, graph, sentence):
    if(id == -1 or id == 0):
        return np.concatenate([
            np.zeros(tags_len),
            np.zeros(glove_size)
        ])
    word_dict = sentence[id - 1]
    tags_hot = np.zeros(tags_len)
    tags_hot[tags[word_dict['upostag']]] = 1
    if 'lemma' in word_dict and word_dict['lemma'] in glove_vectors:
        vector = glove_vectors[word_dict['lemma']]
    else:
        vector = np.zeros(glove_size)
        
    return np.concatenate([tags_hot, vector])
        


# In[22]:


zeros = np.zeros(50)

def representation(stack, queue, sentence, graph):
    stack_pos = get_features(
         stack[-1] if len(stack) > 0 else 0,
         graph,
         sentence
    )
    queue_pos = np.concatenate([
        get_features(
            queue[0] if len(queue) > 0 else 0,
            graph,
            sentence
        ),
        get_features(
            queue[1] if len(queue) > 1 else 0,
            graph,
            sentence
        )
    ])

    if len(stack) > 1 and (stack[-1] in list(graph.nodes())):
        left_stack = get_features(
             (list(graph.successors(stack[-1])))[0] if len(list(graph.successors(stack[-1]))) > 0 else 0,
             graph,
             sentence
         )
        right_stack = get_features(
            (list(graph.successors(stack[-1])))[-1] if len(list(graph.successors(stack[-1]))) > 1 else 0,
             graph,
             sentence
         )

    else:
        left_stack = get_features(
             0,
             graph,
             sentence
         )
        right_stack = get_features(
             0,
             graph,
             sentence
         )
        
    if len(queue) > 0 and (queue[0] in list(graph.nodes())):
        left_queue = get_features(
             (list(graph.successors(queue[0])))[0] if len(list(graph.successors(queue[0]))) > 0 else 0,
             graph,
             sentence
         )
        right_queue = get_features(
            (list(graph.successors(stack[0])))[-1] if len(list(graph.successors(queue[0]))) > 1 else 0,
             graph,
             sentence
         )

    else:
        left_queue = get_features(
             0,
             graph,
             sentence
         )
        right_queue = get_features(
             0,
             graph,
             sentence
        )
    # print len(np.concatenate([stack_pos, queue_pos, right_stack, left_stack, right_queue, left_queue]))  
    return np.concatenate([stack_pos, queue_pos, right_stack, left_stack, right_queue, left_queue])
    


# In[9]:


def load_data(data):
    features = np.array([], dtype=np.float64).reshape(0,469)
    operations = []
    for sentence in data:
        #print "in loop" 
        Rp = []
        vocab = []
        Id = []
        DG = nx.DiGraph()
        for temp2 in sentence:  
            Rp.append([temp2["head"], temp2["id"]])
            vocab.append(temp2["lemma"])
            Id.append(temp2["id"])

        #print Rp
        #print sentence
        #print Id

        feature = []
        stack = [0]
        #print Id
        queue = Id
        operation = []
        feature.append(representation(stack, queue, sentence, DG))
        operation.append(2)
        # get feature vector from stack queue and operation
        # config.append(representation(stack, queue, [])
        stack.append(queue[0])
        queue = queue[1:]
        #for left arc = 0, right arc = 1, shift = 2

        while len(stack) != 1:
            if([stack[-1],stack[-2]]) in Rp:
                # print 'inside if'
                feature.append(representation(stack, queue, sentence, DG))
                DG.add_edge(stack[-1],stack[-2])
                operation.append(0)
                Rp.remove([stack[-1],stack[-2]])
                stack = stack[:-2] + [stack[-1]]
            elif([stack[-2],stack[-1]]) in Rp:
                flag = 0
                for item in Rp:
                    if stack[-1] == item[0]:
                        flag = 1
                        break
                if flag == 0:
                    feature.append(representation(stack, queue, sentence, DG))
                    DG.add_edge(stack[-2],stack[-1])
                    operation.append(1)
                    Rp.remove([stack[-2],stack[-1]])
                    stack = stack[:-1]
                else:
                    feature.append(representation(stack, queue, sentence, DG))
                    operation.append(2)
                    stack.append(queue[0])
                    queue = queue[1:]

            else:
                feature.append(representation(stack, queue, sentence, DG))
                operation.append(2)
                stack.append(queue[0])
                queue = queue[1:]

        operations = np.concatenate([operations,operation])
        feature = np.array(feature)
        features = np.concatenate([features,feature])
        return features, operations
        #print len(feature)


# In[15]:


train_data = load_data_file("/home/saransh/Desktop/data/train.conllu")
print train_data
train_features, train_operations = load_data(train_data)
print train_features, train_operations


# In[23]:


model = MLPClassifier(
    hidden_layer_sizes=model_size,
    random_state=1,
    verbose=True
)


# In[24]:


model.fit(train_features,train_operations)


# In[25]:


test_data = load_data_file("/home/saransh/Desktop/data/test.conllu")


# In[86]:


test_features, test_operations = load_data(test_data)


# In[87]:


test_output = model.predict(test_features)
y_true = test_operations
y_pred = test_output
print accuracy_score(y_true, y_pred)  


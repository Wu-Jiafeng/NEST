from bilmModel import Fine_Tuning_BiLstm_Model_Test
import numpy as np
import os
from BatchGenerator import FineTuningBatchGenerator
from preTrain import Config
import time
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def GetTestInput():
    texts, nodes = {}, {}
    id2embedding = open("../dataset/Graph_Node_Feature_Handled.txt", "r", encoding="utf-8")
    for line in id2embedding.readlines():
        id,embedding = line.split("\t\t")
        embedding = embedding.split(' ')
        texts[id]= np.array(embedding).astype(np.float32).tolist()
    id2embedding.close()

    node2id = open("../dataset/Graph_Origin_Node_ID.txt", "r", encoding="utf-8")
    for line in node2id.readlines():
        node, id = line.strip().split("\t\t")
        nodes[node] = id
    node2id.close()

    GraphURI = open("../dataset/Graph_URI_ID_ESBML.txt", "r", encoding="utf-8")
    lines=GraphURI.readlines()
    ucount,uid=0,None
    for line in lines:
        id = (line.strip().split("\t\t"))[0]
        if uid is None or uid != id:
            uid = id
            ucount+=1
    print(ucount)

    uid=None
    id2graph, id2id, id2feature=[[] for i in range(ucount)],[[] for i in range(ucount)],[[] for i in range(ucount)]
    for line in lines:
        if len(line.strip().split("\t\t")) < 4: continue
        id,subj,pred,obj=line.strip().split("\t\t")
        if uid is None or uid != id: uid=id
        id2graph[int(uid)-101].append([subj, pred, obj])
        id2id[int(uid)-101].append([nodes[subj], nodes[pred], nodes[obj]])
        id2feature[int(uid)-101].append([texts[nodes[subj]], texts[nodes[pred]], texts[nodes[obj]]])
        # 如果ESBMD就uid-1，ESBML就uid-101
    return id2graph,id2id,id2feature

def GetInstance(choose,unchoose,id2feature):
    instance, label = [], []
    tokens_fw, tokens_bw, tokenLabel_fw, tokenLabel_bw = [], [], [], []
    for i in range(len(choose)):
        tokens_fw.append(id2feature[choose[i]])
        tokens_bw.append(id2feature[choose[i]][::-1])
        tokenLabel_fw.append(0)
        tokenLabel_bw.append(0)
    instance.append(np.array(tokens_fw))
    instance.append(np.array(tokens_bw))
    label.append(np.array([np.array(tokenLabel_fw)]).transpose())
    label.append(np.array([np.array(tokenLabel_bw)]).transpose())
    return instance,label

def CalSTSScore(id2feature,config):
    config.is_Diversity = False
    scores=[[] for i in range(len(id2feature))]
    X_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    X_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep, config.input])
    Y_fw = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    Y_bw = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            with tf.variable_scope('lm'):
                lstm_model = Fine_Tuning_BiLstm_Model_Test(config,input=[X_fw,X_bw],label=[Y_fw,Y_bw])
    saver = tf.train.Saver()
    tfConfig = tf.ConfigProto(allow_soft_placement=True)
    tfConfig.gpu_options.allow_growth = True
    with tf.Session(config=tfConfig) as sess:
        checkpoint = tf.train.get_checkpoint_state("../dataset/ckpt1")
        saver.restore(sess, checkpoint.model_checkpoint_path)

        for i in range(len(id2feature)):
            for j in range(len(id2feature[i])):
                choose=[j]
                instance, label = GetInstance(choose,None, id2feature[i])
                feed_dict = {X_fw: instance[0], X_bw: instance[-1], Y_fw: label[0], Y_bw: label[-1]}
                ret = sess.run([lstm_model.loss_input, lstm_model.loss_input_diversity], feed_dict=feed_dict)
                print(ret[0])
                scores[i].append(ret[0][0][0])
        return scores

def GenerateSummary(config):
    output=open("../dataset/ESBMLTop10_","w",encoding="utf-8")
    id2graph, id2id, id2feature = GetTestInput()
    # cal STS score
    scores_sts=CalSTSScore(id2feature,config)
    tf.reset_default_graph()

    #cal DSS score
    config.is_Diversity = True
    X_fw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep*config.topkSize, config.input])
    X_bw = tf.placeholder(dtype=tf.float32, shape=[None, config.TimeStep*config.topkSize, config.input])
    Y_fw = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    Y_bw = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            with tf.variable_scope('lm'):
                lstm_model = Fine_Tuning_BiLstm_Model_Test(config,input=[X_fw,X_bw],label=[Y_fw,Y_bw])

    saver = tf.train.Saver()
    tfConfig = tf.ConfigProto(allow_soft_placement=True)
    tfConfig.gpu_options.allow_growth = True
    with tf.Session(config=tfConfig) as sess:
        checkpoint = tf.train.get_checkpoint_state("../dataset/ckpt1")
        saver.restore(sess, checkpoint.model_checkpoint_path)

        for i in range(len(id2id)):
            # data sample
            last_score,last_choose,p = -float("inf"),random.sample(range(len(id2id[i])),config.topkSize),0.1
            while p>0:
                choose= last_choose
                unchoose=[j for j in range(len(id2id[i])) if j not in choose]
                tri_replaced,tri_replace=random.randint(0,len(choose)-1),random.randint(0,len(unchoose)-1)
                choose[tri_replaced]=unchoose[tri_replace]
                choose_scores=np.array(scores_sts[i])
                score_sts = np.sum(choose_scores[np.array(choose)])
                instance,label=GetInstance(choose,unchoose,id2feature[i])

                # cal DSS score
                instance_div = []
                instance_div.append(np.reshape(instance[0], [1, -1, 300]))
                instance_div.append(np.reshape(instance[-1], [1, -1, 300]))
                feed_dict_div = {X_fw: instance_div[0], X_bw: instance_div[-1], Y_fw: label[0], Y_bw: label[-1]}
                ret = sess.run([lstm_model.loss_input, lstm_model.loss_input_diversity], feed_dict=feed_dict_div)
                score_dss = ret[1][0][0]
                score = score_sts + score_dss
                print(score,last_score)

                if score>=last_score: last_score,last_choose=score,choose
                elif random.random()<p: last_score,last_choose=score,choose

                p-=0.1/len(id2id[i])
            # output
            for index in last_choose: output.write("\t".join(id2graph[i][index])+"\n")
    output.close()

if __name__=="__main__":
    #GetTestInput()
    FTBG = FineTuningBatchGenerator()
    config = Config(learning_rate=0.2, batchsize=1, input=300, timestep=3, projection_dim=300,epoch=1, hidden_unit=4096,
                    n_negative_samples_batch=8192, token_size=0, is_Training=False,topkSize=10)
    GenerateSummary(config)
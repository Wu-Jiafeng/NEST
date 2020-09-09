import numpy as np
from gensim.models import FastText
import re
import os
import random
from rdflib.graph import Graph
from rdflib import URIRef
from rfc3987 import match
from nltk.corpus import stopwords

pattern = r'''(?x)                     # set flag to allow verbose regexps
        (?:[A-Z]\.)+                   # abbreviations, e.g. U.S.A.
        |\w+(?:\'t)                    # the negative
        |\w+(?:-\w+)*                  # words with optional internal hyphens
        |\$?\d+(?:\.\d+)?%?            # currency and percentages, e.g. $12.40, 82%
        #|\w+(?:\.\w+)*@\w+(?:\.\w+)*  # Email address?
        #|(?:\'d)|(?:\'s)|(?:\'m)|(?:\'re)|(?:\'ll)|(?:\'ve) # abbr
        #|\.\.\.                       # ellipsis
        #|(?:[.,;"'?():-_`])           # these are separate tokens; includes ], [
     '''

def DownSampleGraph():
    dg=open("../dataset/ds_longabstract_en.nt","w",encoding="utf-8")
    g=open("../dataset/longabstract_en.nt","r",encoding="utf-8")
    for line in g.readlines():
        if re.search("\"\"@en",line) is not None: continue
        if random.random()<0.005: dg.write(line)
    dg.close()
    g.close()

def GetEntityAbstract():
    EntityAbstract=open("../dataset/Entity_Abstract.txt","w",encoding="utf-8")
    g = Graph()
    g.parse("../dataset/ds_longabstract_en.nt", format="nt")
    for subj, _, obj in g: EntityAbstract.write(subj + "\t\t" + obj.replace('\n',' ') + "\n")
    EntityAbstract.close()

def GetSplitFdata():
    EntityAbstract = open("../dataset/Entity_Abstract.txt", "r", encoding="utf-8")
    count=1
    for line in EntityAbstract.readlines():
        print(count)
        line=line.strip().split("\t\t")
        g = Graph()
        g.parse(line[0])
        try: g.serialize("../dataset/FinetuneData/"+str(count)+"_desc.nt",format="nt")
        except: pass
        count+=1
    EntityAbstract.close()

def GetGraphURI(): # 标注ID和无标注的都在这里
    GraphURI = open("../dataset/FGraph_URI.txt", "w", encoding="utf-8")
    fdatapath="../dataset/FinetuneData/"
    count=1
    for _, _, files in os.walk(fdatapath):
        for f in files:
            print(count)
            if count > 2000: break
            g=Graph()
            g.parse(fdatapath+f,format="nt")
            for subj, pred, obj in g:
                print(subj, pred, obj)
                #GraphURI.write(f.replace("_desc.nt","")+"\t\t"+subj + "\t\t" + pred + "\t\t" + obj.strip().replace('\n', '').replace('\r', '') + "\n")
                GraphURI.write(subj + "\t\t" + pred + "\t\t" + obj.strip().replace('\n', '').replace('\r', '') + "\n")
            count += 1
    GraphURI.close()

def GetNodeID():
    Nodes=set()
    GraphURI = open("../dataset/FGraph_URI.txt", "r", encoding="utf-8")
    for line in GraphURI.readlines():
        for node in line.strip().split("\t\t"): Nodes.add(node)
    GraphURI.close()

    Nodes=list(Nodes)
    NodeID=open("../dataset/FGraph_Origin_Node_ID.txt", "w", encoding="utf-8")
    for i in range(len(Nodes)): NodeID.write(Nodes[i]+"\t\t"+str(i)+"\n")
    NodeID.close()

def ExtractTextFeatures():
    id2text=open("../dataset/FGraph_ID_Text.txt", "a", encoding="utf-8")
    NodeID=open("../dataset/FGraph_Origin_Node_ID.txt", "r", encoding="utf-8")
    count=0
    for line in NodeID.readlines():
        if count<36661:
            count+=1
            continue
        if count>=81000: break
        uri,id=line.strip().split("\t\t")
        text=""
        if match(uri, rule='IRI_reference') is None: text=uri
        elif len(uri.split("#"))>1: text=re.sub(r"(\w)([A-Z])", r"\1 \2", (uri.split("#"))[-1])
        else:
            g = Graph()
            try:
                g.parse(uri)
                text=g.label(URIRef(uri))
            except: print("不能解析！")
            if text=="":
                uri_s1=uri.split("resource/")
                if len(uri_s1)>1: text=uri_s1[-1]
                else:
                    uri_s2=(uri.split("/"))[-1]
                    if uri_s2 != "": text=uri_s2
                    else: text=uri
        id2text.write(id + "\t\t" + text + "\n")
        print(count)
        count+=1
    id2text.close()
    NodeID.close()

def GetTextTokens():
    text2tokens=open("../dataset/FGraph_Text_Tokens.txt", "w", encoding="utf-8")
    id2text = open("../dataset/FGraph_ID_Text.txt", "r", encoding="utf-8")
    for line in id2text.readlines():
        id, text = line.strip().split("\t\t")
        value = re.sub(r'(?:\'d)|(?:\'s)|(?:\'m)|(?:\'re)|(?:\'ll)|(?:\'ve)|(?:_)+', '', text.lower())  # replace abbr
        tokens = re.findall(pattern, value)
        tokens = [re.sub(r'\d+', 'N', token) for token in tokens if token not in stopwords.words('english')]
        text2tokens.write("\t\t".join(tokens)+"\n")
    text2tokens.close()
    id2text.close()
    print("Texts have been tokenized!")

def GetFasttextFeature():
    tokenss,ftokenss=[],[]
    text2tokens = open("../dataset/Graph_Text_Tokens.txt", "r", encoding="utf-8")
    ftext2tokens = open("../dataset/FGraph_Text_Tokens.txt", "r", encoding="utf-8")
    for line in text2tokens.readlines(): tokenss.append(line.strip().split("\t\t"))
    for line in ftext2tokens.readlines(): ftokenss.append(line.strip().split("\t\t"))
    text2tokens.close()
    ftext2tokens.close()

    print("Now start FastText training!")
    model = FastText(tokenss+ftokenss, size=300,window=3, min_count=1, iter=10,word_ngrams=0)
    model.save("../dataset/FastText/ft.model")

    print("Now calculate FastText features!")
    id2feature=open("../dataset/Graph_Node_Feature_Handled.txt", "w", encoding="utf-8")
    fid2feature = open("../dataset/FGraph_Node_Feature_Handled.txt", "w", encoding="utf-8")
    for i in range(len(tokenss)):
        feature=np.zeros(300,dtype=np.float32)
        for token in tokenss[i]: feature+=model.wv[token]
        feature/=len(tokenss[i])
        id2feature.write(str(i)+"\t\t"+" ".join([str(i) for i in feature.tolist()])+"\n")
    id2feature.close()
    for i in range(len(ftokenss)):
        feature=np.zeros(300,dtype=np.float32)
        for token in ftokenss[i]: feature+=model.wv[token]
        feature/=len(ftokenss[i])
        fid2feature.write(str(i)+"\t\t"+" ".join([str(i) for i in feature.tolist()])+"\n")
    fid2feature.close()
    print("Done!")

if __name__=="__main__":
    #DownSampleGraph()
    #GetEntityAbstract()
    #GetSplitFdata()
    GetGraphURI()
    #GetNodeID()
    #ExtractTextFeatures()
    #GetTextTokens()
    #GetFasttextFeature()
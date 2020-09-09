import numpy as np
from gensim.models import FastText
import re
import os
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

def GetEList():
    uri2label={}
    elist=open("../dataset/ESBM-master/v1.0/ESBM_benchmark/elist.txt","r",encoding="utf-8")
    lines=(elist.readlines())[1:]
    for line in lines:
        eid, eclass, euri, elabel, triplenum = line.strip().split("\t")
        uri2label[euri] = elabel
    elist.close()
    return uri2label

def GetGraphURI():
    GraphURI=open("../dataset/Graph_URI.txt","w",encoding="utf-8")
    datapath="../dataset/ESBM-master/v1.0/ESBM_benchmark/"
    for subdata in ["dbpedia/","lmdb/"]:
        for sroot, sdirs, sfiles in os.walk(datapath+subdata):
            for sdir in sdirs:
                g=Graph()
                g.parse(sroot+sdir+"/"+sdir+"_desc.nt", format="nt")
                for subj, pred, obj in g:
                    print(subj, pred, obj)
                    #GraphURI.write(str(sdir) + "\t\t" + subj + "\t\t" + pred + "\t\t" + obj + "\n")
                    GraphURI.write(subj+"\t\t"+pred+"\t\t"+obj+"\n")
    GraphURI.close()

def GetNodeID():
    Nodes=set()
    GraphURI = open("../dataset/Graph_URI.txt", "r", encoding="utf-8")
    for line in GraphURI.readlines():
        for node in line.strip().split("\t\t"): Nodes.add(node)
    GraphURI.close()

    Nodes=list(Nodes)
    NodeID=open("../dataset/Graph_Origin_Node_ID.txt", "w", encoding="utf-8")
    for i in range(len(Nodes)): NodeID.write(Nodes[i]+"\t\t"+str(i)+"\n")
    NodeID.close()

def ExtractTextFeatures():
    uri2label=GetEList()
    id2text=open("../dataset/Graph_ID_Text.txt", "w", encoding="utf-8")
    NodeID=open("../dataset/Graph_Origin_Node_ID.txt", "r", encoding="utf-8")
    count=0
    for line in NodeID.readlines():
        uri,id=line.strip().split("\t\t")
        text=""
        if match(uri, rule='IRI_reference') is None: text=uri
        elif uri in uri2label.keys(): text=uri2label[uri]
        elif len(uri.split("#"))>1: text=re.sub(r"(\w)([A-Z])", r"\1 \2", (uri.split("#"))[-1])
        else:
            g = Graph()
            try:
                g.parse(uri)
                text=g.label(URIRef(uri))
            except: print("不能解析！")
            if text=="":
                uri_s=uri.split("resource/")
                if len(uri_s)>1: text=uri_s[-1]
                else: text=(uri.split("/"))[-1]
        id2text.write(id + "\t\t" + text + "\n")
        print(count)
        count+=1
    id2text.close()
    NodeID.close()

def GetTextTokens():
    text2tokens=open("../dataset/Graph_Text_Tokens.txt", "w", encoding="utf-8")
    id2text = open("../dataset/Graph_ID_Text.txt", "r", encoding="utf-8")
    for line in id2text.readlines():
        id, text = line.strip().split("\t\t")
        value = re.sub(r'(?:\'d)|(?:\'s)|(?:\'m)|(?:\'re)|(?:\'ll)|(?:\'ve)|(?:_)+', '', text.lower())  # replace abbr
        tokens = re.findall(pattern, value)
        tokens = [re.sub(r'\d+', 'N', token) for token in tokens if token not in stopwords.words('english')]
        text2tokens.write("\t\t".join(tokens)+"\n")
    text2tokens.close()
    id2text.close()
    print("Texts have been tokenized!")

if __name__=="__main__":
    #GetNodeID()
    GetGraphURI()
    #readdata()
    #ExtractTextFeatures()
    #GetTextTokens()
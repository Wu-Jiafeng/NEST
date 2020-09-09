from rdflib.graph import Graph
from nltk.corpus import stopwords
import Levenshtein
import re

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
month2num={"january": "01","feburary":"02","march":"03","april":"04","may":"05","june":"06",
           "july":"07","august":"08","september":"09","october":"10","november":"11","december":"12",
           "jan": "01","feb":"02","mar":"03","apr":"04","jun":"06",
           "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}

def TextNormalize(text):
    tokens_n=[]
    value = re.sub(r'(?:\'d)|(?:\'s)|(?:\'m)|(?:\'re)|(?:\'ll)|(?:\'ve)|(?:_)+', '', text.lower())  # replace abbr
    tokens = re.findall(pattern, value)
    for token in tokens:
        if token in stopwords.words('english'): continue
        if token in month2num.keys(): tokens_n.append(month2num[token])
        else: tokens_n.append(token)
    return tokens_n

def match(text,abstract,mchar=7,edis=3):
    match_count,nflag=0,True
    for word in text:
        if not (re.search(r'\d', word) and word in abstract): nflag=False #都是出现在摘要里的数字
        if len(word)>mchar: #编辑距离
            for cword in abstract:
                if Levenshtein.distance(word,cword)<edis:
                    match_count+=1
                    break
        if match_count>=0.5*len(text): return "1"
    if nflag: return "1"
    else: return "0"

def GetLabel():
    GraphURI_labeled=open("../dataset/FGraph_URI_labeled.txt","w",encoding="utf-8")
    abstracts,texts,nodes=[],{},{}

    EntityAbstract=open("../dataset/Entity_Abstract.txt","r",encoding="utf-8")
    for line in EntityAbstract.readlines(): abstracts.append((line.strip().split("\t\t"))[-1])
    EntityAbstract.close()

    id2text=open("../dataset/FGraph_ID_Text.txt","r",encoding="utf-8")
    for line in id2text.readlines():
        id,text=line.strip().split("\t\t")
        texts[id]=text
    id2text.close()

    node2id=open("../dataset/FGraph_Origin_Node_ID.txt","r",encoding="utf-8")
    for line in node2id.readlines():
        node,id=line.strip().split("\t\t")
        nodes[node]=id
    node2id.close()

    GraphURI=open("../dataset/FGraph_URI_ID.txt","r",encoding="utf-8")
    uid,uabstract=None,None
    for line in GraphURI.readlines():
        fid,subj,pred,obj=line.strip().split("\t\t")
        print(fid)
        if uid is None or uid!=fid:
            uid=fid
            uabstract=TextNormalize(abstracts[int(uid)-1])
        text=TextNormalize(texts[nodes[obj]])
        if text=="":label="0"
        else: label=match(text,uabstract)
        GraphURI_labeled.write(subj+"\t\t"+pred+"\t\t"+obj+"\t\t"+label+"\n")
    GraphURI.close()
    GraphURI_labeled.close()


if __name__=="__main__":
    GetLabel()
import os
from rdflib.graph import Graph


def bulidbenchmark():
    f1 = open("../dataset/benchmarkESBML.txt", "w", encoding="utf-8")
    datapath = "../dataset/ESBM-master/v1.0/ESBM_benchmark/"
    for subdata in ["lmdb/"]:
        for sroot, sdirs, sfiles in os.walk(datapath + subdata):
            for sdir in sdirs:
                vocab = {}
                g = Graph()
                g.parse(sroot + sdir + "/" + sdir + "_desc.nt", format="nt")
                for subj, pred, obj in g: vocab[subj+"\t"+pred+"\t"+obj]=[]
                count=0
                for i in range(6):
                    sg=Graph()
                    sg.parse(sroot + sdir + "/" + sdir + "_gold_top5_"+str(i)+".nt", format="nt")
                    for subj, pred, obj in sg: vocab[subj+"\t"+pred+"\t"+obj].append("1")
                    for key in vocab.keys():
                        if len(vocab[key])==count: vocab[key].append("0")
                    count+=1
                for i in range(6):
                    sg=Graph()
                    sg.parse(sroot + sdir + "/" + sdir + "_gold_top10_"+str(i)+".nt", format="nt")
                    for subj, pred, obj in sg: vocab[subj+"\t"+pred+"\t"+obj].append("1")
                    for key in vocab.keys():
                        if len(vocab[key])==count: vocab[key].append("0")
                    count+=1
                for key in vocab.keys(): f1.write(key+"\t\t"+" ".join(vocab[key])+"\n")
    f1.close()

if __name__=="__main__":
    #bulidbenchmark()
    uri2score={}
    f1=open("../dataset/benchmarkESBML.txt","r",encoding="utf-8").readlines()
    f2=open("../dataset/ESBMLTop10_","r",encoding="utf-8").readlines()
    f3=open("../dataset/ESBMLTop10","w",encoding="utf-8")
    for line in f1:
        uri,score=line.strip().split("\t\t")
        uri2score[uri]=score
    for line in f2:
        f3.write(line.strip()+"\t\t"+uri2score[line.strip()]+"\n")
    f3.close()


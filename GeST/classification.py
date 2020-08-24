import sys
from sklearn.feature_selection import mutual_info_classif,chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from statistics import mean,stdev
from xgboost import XGBClassifier
import numpy,pickle,csv
from sklearn.svm import SVC

def _max_silhouette(csv):
    # checking for image change in CSV file
    prev_image,image="",""
    silhouette,pri=[],[]
    with open(csv, "r") as f:
        for i,line in enumerate(f):
            scores=line.split(";")
            image=scores[0]
            if(image != prev_image and prev_image != ""):
                index_max_sil,max_sil = silhouette.index(max(silhouette, key=lambda x: x[0])), max(silhouette, key=lambda x: x[0])
                pri.append(silhouette[index_max_sil][1])
                print(max_sil, silhouette[index_max_sil][1])
                silhouette=[]
            # couple silhouette<->PRI
            silhouette.append((float(scores[4]),float(scores[2])))
            prev_image = image
    index_max_sil,max_sil = silhouette.index(max(silhouette, key=lambda x: x[0])), max(silhouette, key=lambda x: x[0])
    pri.append(silhouette[index_max_sil][1])
    print(max_sil, silhouette[index_max_sil][1])
    print(len(pri),"mean and stdev",mean(pri),stdev(pri))

def _min_daviesbouldin(csv):
    # checking for image change in CSV file
    prev_image,image="",""
    daviesbouldin,pri=[],[]
    with open(csv, "r") as f:
        for line in f:
            scores=line.split(";")
            image=scores[0]
            # couple silhouette<->PRI
            if(image != prev_image and prev_image != ""):
                index_max_db,max_db = daviesbouldin.index(min(daviesbouldin, key=lambda x: x[0])), min(daviesbouldin, key=lambda x: x[0])
                pri.append(daviesbouldin[index_max_db][1])
                print(max_db, daviesbouldin[index_max_db][1])
                daviesbouldin=[]
            daviesbouldin.append((float(scores[5]),float(scores[2])))
            prev_image = image
    index_max_db,max_db = daviesbouldin.index(min(daviesbouldin, key=lambda x: x[0])), min(daviesbouldin, key=lambda x: x[0])
    pri.append(daviesbouldin[index_max_db][1])
    print(max_db, daviesbouldin[index_max_db][1])
    print(len(pri),"mean and stdev",mean(pri),stdev(pri))

def _loadscores_traintest(csvfile):
    fieldnames = ["image","PRE","ARE","MSE","VOI","modregions","silemb","silfeatcolor","silfeathog","dbemb","dbfeat_color","db_feat_hog","chemb","chfeat_color","chfeat_hog","n_clusters","mean_n_clusters","stdev_n_clusters","min_weight_Gc","max_weight_Gc","mean_weight_Gc"]
    # checking for image change in CSV file
    prev_image,image="",""
    data, pri = [],[]
    target = []
    avg = [ [] for i in range(40) ]
    #target = [0]*lines
    begin=0
    max_pri=[]
    names=""
    to_remove= []
    with open(csvfile, newline='') as f:
        reader = csv.DictReader(f,fieldnames=fieldnames,delimiter=";")
        # FIXME: EXTRACT NAMES FROM FIRST LINE
        for i,scores in enumerate(reader):
            image=scores["image"].split(".")[0]
            if(not "GT" in scores["image"]):
                pri.append(float(scores["PRI"]))
                avg[int(scores["n_clusters"])].append(float(scores["PRI"]))
                #target.append(0)
            #else:
            #    target.append(1)
            #target[i]= 1 if "GT" in scores[0] else 0
            data.append(list(map(float,list(scores.values())[1:])))
            if(image != prev_image and prev_image != ""):
                index_best,best = pri[:-1].index(max(pri[:-1])),max(pri[:-1])
                # we review everything we just saw:
                '''target.extend([0 for l in range(len(pri[:-1]))])
                for pos in range(begin,i-1):
                    # is PRI far away?
                    if data[pos][0] >= best - 0.001:
                        target[pos]=1
                    elif data[pos][0] < best - 0.1 and data[pos][0] > best - 0.5:
                        target[pos]=0
                    else:
                        # far away from optimal BUT still pretty close: we remove
                        to_remove.append(pos)'''
                # NOTE: the best may be considered as minimum mean of all error scores?
                _extend = [1 if pri[l] >= best-0.06 else 0 for l in range(len(pri[:-1]))]
                '''pos_max=numpy.argpartition(pri[:-1],int(0.4*-len(pri[:-1])))[int(0.4*-len(pri[:-1])):]
                _extend = [1 if l in pos_max else 0 for l in range(len(pri[:-1]))]'''
                
                target.extend(_extend)
                
                max_pri.append(max(pri[:-1]))
                pri=[pri[-1]]
                begin=i
            # FIXME: remove when too close to optimal
            prev_image = image
    # emptying buffer one last time
    #target[begin+index_best]=1
    max_pri.append(max(pri))
    index_best,best = pri.index(max(pri)),max(pri)
    '''target.extend([0 for l in range(len(pri))])
    for pos in range(begin,i):
        # is PRI far away?
        if data[pos][0] >= best - 0.001:
            target[pos]=1
        elif data[pos][0] < best - 0.1 and data[pos][0] > best - 0.5:
            target[pos]=0
        else:
            # far away from optimal BUT still pretty close: we remove
            to_remove.append(pos)
    data = [d for i,d in enumerate(data) if i not in to_remove]
    target = [t for i,t in enumerate(target) if i not in to_remove]'''
    
    '''pos_max=numpy.argpartition(pri[:-1],int(0.4*-len(pri[:-1])))[int(0.4*-len(pri[:-1])):]
    _extend = [1 if l in pos_max else 0 for l in range(len(pri))]'''
    _extend = [1 if pri[l] >= best-0.06 else 0 for l in range(len(pri))]
    target.extend(_extend)
    print("max possible", len(max_pri),mean(max_pri))
    return numpy.asarray(data),numpy.asarray(target),fieldnames[2:],avg

# FIXME: should only contain 1 for groundtruth partitions (but how to describe them?)
def _loadscores_val(csvfile):
    fieldnames = ["image","PRI","ARI","MSE","VOI","modregions","silemb","silfeatcolor","silfeathog","dbemb","dbfeat_color","db_feat_hog","chemb","chfeat_color","chfeat_hog","n_clusters","mean_n_clusters","stdev_n_clusters","min_weight_Gc","max_weight_Gc","mean_weight_Gc"]
    # checking for image change in CSV file
    data = []
    prev_image,image="",""
    slices = []
    begin=0
    max_pri,pri=[],[]
    avg = [ [] for i in range(40) ]
    with open(csvfile, newline='') as f:
        reader = csv.DictReader(f,fieldnames=fieldnames,delimiter=";")
        # FIXME: EXTRACT NAMES FROM FIRST LINE
        for i,scores in enumerate(reader):
            avg[int(scores["n_clusters"])].append(float(scores["PRI"]))
            # first two columns = size k-means and name
            image=scores["image"]
            pri.append(float(scores["PRI"]))
            if(image != prev_image and prev_image != ""):
                slices.append((begin,i-1))
                max_pri.append((begin+pri.index(max(pri[:-1])),max(pri[:-1])))
                begin=i
                pri=[pri[-1]]
            data.append(scores)
            prev_image = image
    # emptying buffer one last time
    max_pri.append((begin+pri.index(max(pri)),max(pri)))
    slices.append((begin,i))
    print("max possible for val data:",mean([e[1] for e in max_pri]),len(max_pri))
    data = [{k: float(v) for k,v in d.items()} for d in data]
    return numpy.asarray(data),slices,fieldnames,avg,max_pri

if __name__=="__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    '''print("silhouette heuristic")
    _max_silhouette(sys.argv[1])
    print("davies bouldin heuristic")
    _min_daviesbouldin(sys.argv[1])'''

    X_train,y_train,names,avg = _loadscores_traintest(sys.argv[1])
    # getting rid of PRI column
    X_train = numpy.delete(X_train,0,axis=1)
    print(len([i for i in y_train if i == 1])/len(y_train))

    pdy = pd.DataFrame(y_train)
    pdy.columns=['target']

    pdX = pd.DataFrame(X_train)
    pdX.columns=names

    df = pd.concat([pdX, pdy], axis = 1)
    print(df)

    ax = sns.heatmap(df.corr(), annot = True)
    plt.show()

    res=mutual_info_classif(X_train,y_train)
    print(res)

    #model = LogisticRegression()
    model = XGBClassifier()
    model_svc=SVC()
    model.fit(X_train, y_train) # On apprend le modèle
    model_svc.fit(X_train, y_train) # On apprend le modèle

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    #pickle.dump(model, open("model_logistic_regression.pkl", 'wb'))
    #loaded_model = pickle.load(open("model_logistic_regression.pkl", 'rb'))

    all_values,slices,all_names,avg,max_pri = _loadscores_val(sys.argv[2])
    X_val=numpy.asarray([list(d.values())[2:] for d in all_values])
    
    print([(mean(l),stdev(l)) if l != [] else 0 for l in avg])
    plt.errorbar(list(range(40)), [mean(l) if l != [] else 0 for l in avg ], [stdev(l) if l != [] else 0 for l in avg],linestyle="None", marker="^")
    plt.xticks(list(range(40)))
    plt.show()
    
    pdX = pd.DataFrame([list(d.values()) for d in all_values])
    pdX.columns=all_names
    print(pdX)
    
    pdX = pd.DataFrame(X_val)
    pdX.columns=names
    print(pdX)

    # to remove PRI
    y_pred_proba=model.predict_proba(X_val)
    y_pred=model_svc.predict(X_val)

    mod_predicted = []
    sil_predicted = []
    db_predicted = []
    proba_predicted = []
    mean_predicted = []
    max_predicted = []
    n_clusters = []
    dsc = []
    aupif=0
    for index,s in enumerate(slices):
        slice_proba = y_pred_proba[s[0]:s[1]+1].tolist()
        best_proba = slice_proba.index(max(slice_proba,key=lambda x: x[1]))
        best_predict = [all_values[p]["PRI"] for p in range(s[0],s[1]+1)]
        max_predicted.append(max(best_predict))
        all_predicted = [i for i in range(s[0],s[1]+1) if y_pred[i] == 1]
        # all predicted partition in this slice
        if(all_predicted != []):
            modularities = [all_values[i]["modregions"] for i in all_predicted]
            silhouette = [all_values[i]["silemb"] for i in all_predicted]
            daviesbouldin = [all_values[i]["dbemb"] for i in all_predicted]
            nclusters = [all_values[i]["n_clusters"] for i in all_predicted]
            meandsc = [mean([all_values[i]["modregions"],all_values[i]["silemb"],1./all_values[i]["dbemb"]]) for i in all_predicted]
            mod_predicted.append((all_values[all_predicted[modularities.index(max(modularities))]]["n_clusters"],all_predicted[modularities.index(max(modularities))]))
            sil_predicted.append(all_predicted[silhouette.index(max(silhouette))])
            db_predicted.append(all_predicted[daviesbouldin.index(min(daviesbouldin))])
            n_clusters.append(all_predicted[nclusters.index(max(nclusters))])
            dsc.append(all_predicted[meandsc.index(max(meandsc))])
            #meandbsil = [mean([X_val[i][2],X_val[i][3]]) for i in all_predicted]
        else:
            aupif+=1
            modularities = [all_values[i]["modregions"] for i in range(s[0],s[1]+1)]
            silhouette = [all_values[i]["silemb"] for i in range(s[0],s[1]+1)]
            daviesbouldin  = [all_values[i]["dbemb"] for i in range(s[0],s[1]+1)]
            nclusters = [all_values[i]["n_clusters"] for i in range(s[0],s[1]+1)]
            meandsc = [mean([all_values[i]["modregions"],all_values[i]["silemb"],1./all_values[i]["dbemb"]]) for i in range(s[0],s[1]+1)]
            mod_predicted.append((all_values[s[0]+modularities.index(max(modularities))]["n_clusters"],s[0]+modularities.index(max(modularities))))
            sil_predicted.append(s[0]+silhouette.index(max(silhouette)))
            db_predicted.append(s[0]+daviesbouldin.index(min(daviesbouldin)))
            n_clusters.append(s[0]+nclusters.index(max(nclusters)))
            dsc.append(s[0]+meandsc.index(max(meandsc)))
        #print(all_predicted)
        #print(modularities)
        #mean_predicted.append(all_predicted[meandbsil.index(max(meandbsil))])
        if(slice_proba[best_proba][1]) < 0.35:
            if all_predicted != []:
                modularities=[all_values[i]["silemb"] for i in all_predicted]
            else:                
                modularities = [all_values[i]["silemb"] for i in range(s[0],s[1]+1)]
            proba_predicted.append(s[0]+modularities.index(max(modularities)))
        else:
            proba_predicted.append(s[0]+best_proba)

    print("AU PIF",aupif)
    mods=[all_values[i[1]]["PRI"] for i in mod_predicted]
    sils=[all_values[i]["PRI"] for i in sil_predicted]
    probas=[all_values[i]["PRI"] for i in proba_predicted]
    dbs=[all_values[i]["PRI"] for i in db_predicted]
    dbsil=[all_values[i]["PRI"] for i in mean_predicted]
    ncmax=[all_values[i]["PRI"] for i in n_clusters]
    dscmax=[all_values[i]["PRI"] for i in dsc]
    print([i[1] for i in mod_predicted])
    #print(list((i,j) for (i,j) in list(zip(proba_predicted,probas))))
    print("{} modularity: {}".format(len(mods),mean(mods)))
    print("silhouette: {}".format(mean(sils)))
    print("davies bouldin: {}".format(mean(dbs)))
    print("nclusters: {}".format(mean(ncmax)))
    print("proba: {}".format(mean(probas)))
    print("mean: {}".format(mean(dscmax)))
    print("{} max {}".format(len(max_predicted),mean(max_predicted)))

    model_scaled = XGBClassifier()
    model_scaled.fit(X_train_scaled,y_train)

    X_val_scaled = scaler.fit_transform(X_val)
    pdX = pd.DataFrame(X_val_scaled)
    pdX.columns=names
    print(pdX.head())

    y_pred=model_scaled.predict(X_val_scaled)
    print(len([i for i in y_pred if i == 1])/len(y_pred))
    y_pred_proba=model_scaled.predict_proba(X_val_scaled)

    mod_predicted = []
    sil_predicted = []
    db_predicted = []
    proba_predicted = []
    mean_predicted = []
    max_predicted = []
    n_clusters = []
    dsc = []
    aupif=0
    for index,s in enumerate(slices):
        slice_proba = y_pred_proba[s[0]:s[1]+1].tolist()
        best_proba = slice_proba.index(max(slice_proba,key=lambda x: x[1]))
        best_predict = [all_values[p]["PRI"] for p in range(s[0],s[1]+1)]
        max_predicted.append(max(best_predict))
        all_predicted = [i for i in range(s[0],s[1]+1) if y_pred[i] == 1]
        # all predicted partition in this slice
        if(all_predicted != []):
            modularities = [all_values[i]["modregions"] for i in all_predicted]
            silhouette = [all_values[i]["silemb"] for i in all_predicted]
            daviesbouldin = [all_values[i]["dbemb"] for i in all_predicted]
            nclusters = [all_values[i]["n_clusters"] for i in all_predicted]
            meandsc = [mean([all_values[i]["modregions"],all_values[i]["silemb"],1./all_values[i]["dbemb"]]) for i in all_predicted]
            mod_predicted.append(all_predicted[modularities.index(max(modularities))])
            sil_predicted.append(all_predicted[silhouette.index(max(silhouette))])
            db_predicted.append(all_predicted[daviesbouldin.index(min(daviesbouldin))])
            n_clusters.append(all_predicted[nclusters.index(max(nclusters))])
            dsc.append(all_predicted[meandsc.index(max(meandsc))])
            #meandbsil = [mean([X_val[i][2],X_val[i][3]]) for i in all_predicted]
        else:
            #aupif+=1
            modularities = [all_values[i]["modregions"] for i in range(s[0],s[1]+1)]
            silhouette = [all_values[i]["silemb"] for i in range(s[0],s[1]+1)]
            daviesbouldin  = [all_values[i]["dbemb"] for i in range(s[0],s[1]+1)]
            nclusters = [all_values[i]["n_clusters"] for i in range(s[0],s[1]+1)]
            meandsc = [mean([all_values[i]["modregions"],all_values[i]["silemb"],1./all_values[i]["dbemb"]]) for i in range(s[0],s[1]+1)]
            mod_predicted.append(s[0]+modularities.index(max(modularities)))
            sil_predicted.append(s[0]+silhouette.index(max(silhouette)))
            db_predicted.append(s[0]+daviesbouldin.index(min(daviesbouldin)))
            n_clusters.append(s[0]+nclusters.index(max(nclusters)))
            dsc.append(s[0]+meandsc.index(max(meandsc)))
        #print(all_predicted)
        #print(modularities)
        #mean_predicted.append(all_predicted[meandbsil.index(max(meandbsil))])
        if(slice_proba[best_proba][1]) < 0.35:
            if all_predicted != []:
                modularities=[all_values[i]["silemb"] for i in all_predicted]
            else:                
                modularities = [all_values[i]["silemb"] for i in range(s[0],s[1]+1)]
            proba_predicted.append(s[0]+modularities.index(max(modularities)))
        else:
            proba_predicted.append(s[0]+best_proba)

    print("AU PIF",aupif)
    mods=[all_values[i]["PRI"] for i in mod_predicted]
    sils=[all_values[i]["PRI"] for i in sil_predicted]
    probas=[all_values[i]["PRI"] for i in proba_predicted]
    dbs=[all_values[i]["PRI"] for i in db_predicted]
    dbsil=[all_values[i]["PRI"] for i in mean_predicted]
    ncmax=[all_values[i]["PRI"] for i in n_clusters]
    dscmax=[all_values[i]["PRI"] for i in dsc]
    print([i for i in mod_predicted])
    #print(list((i,j) for (i,j) in list(zip(mod_predicted,mods))))
    #print(list((i,j) for (i,j) in list(zip(proba_predicted,probas))))
    print("{} modularity: {}".format(len(mods),mean(mods)))
    print("silhouette: {}".format(mean(sils)))
    print("davies bouldin: {}".format(mean(dbs)))
    print("proba: {}".format(mean(probas)))
    print("nclusters: {}".format(mean(ncmax)))
    print("mean: {}".format(mean(dscmax)))
    print("{} max {}".format(len(max_predicted),mean(max_predicted)))

    # next we should use this on validation, and then compute PRI for everything predicted bu the model
    # and validation features should be computed WITHOUT PRI, hence pretty fast

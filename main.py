from pandas import read_csv, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import random
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans, kmeans2,vq

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])

def find_centers(X, K):
    print(X)
    # Initialize to K random centers
    oldmu = random.sample(list(X), K)
    mu = random.sample(list(X), K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return (mu, clusters)

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X



def quebra_secoes():
    dados = read_csv("locais_com_votos.csv")
    colunas = dados.columns
    saida = {}
    for c in colunas:
        saida[c] = {}
    saida["secao"] = {}
    dados = dados.to_dict()
    index = 0
    for i in dados["secoes_eleitorais"]:
        secoes = acha_secoes(dados["secoes_eleitorais"][i])
        for j in secoes:
            for key in dados:
                saida[key][index] = dados[key][i]
            saida["secao"][index] = j
            index += 1

    final = DataFrame(saida)
    final["id"] = final.apply(lambda t:str(t["zona_eleitoral_nro"])+"-"+str(t["secao"]),axis=1)
    final.to_csv("locais_com_votacao_trabalhada.csv",index=False)
    return final

def acha_secoes(texto):
    itens = texto.replace("ª","").replace(".","").strip().split(";")
    saida = []
    for i in itens:
        if i.strip().isdigit():
            saida.append(int(i))
        elif "à" in i:
            temp = i.replace("Da ","").replace(" ","").split("à")
            temp = [int(i) for i in temp]
            temp = range(temp[0],temp[1]+1)
            for j in temp:
                saida.append(j)

    return saida

def plota_coordenadas():
    dados = read_csv("secoes_com_votacao.csv")
    #divisor = "bairro"
    #cmap = plt.get_cmap('jet')
    #colors = itertools.cycle([cmap(i) for i in np.linspace(0, 1, len(set(dados[divisor])))])
    #for z in dados[divisor]:
    #    subdados = dados[dados[divisor] == z]
    #    plt.scatter(subdados.lat,subdados.long, color=next(colors))
        #subdados.plot(x='lat', y='long', kind='scatter', color=next(colors))
    #plt.show()

    # data generation
    #data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))
    dados["latlongaptos"] = dados.apply(lambda t:(t["lat"],t["long"],t["aptos_por_local"]),axis=1)
    dados["latlong"] = dados.apply(lambda t:(t["lat"],t["long"]),axis=1)
    pontos = list(dados["latlongaptos"])
    data = vstack(pontos)

    num_iteracoes = 0
    num_clusters = 6

    while True:
        num_iteracoes +=1
        if num_iteracoes % 50 == 0:
            print("ITERAÇÃO NÚMERO "+str(num_iteracoes))

        #vamos tentar ir dividindo nossos clusteres por 2 várias vezes
        #calculos usando scipy
        centroids = find_centers(data,num_clusters)
        centroids = np.array([t for t in centroids[0]])
        idx,_ = vq(data,centroids)

        #garante que temos 55 clusters
        if (len(centroids)) != num_clusters:
            continue

        grupos = {}
        for k in range(num_clusters):
            temp1 = list(data[idx==k,0])
            temp2 = list(data[idx==k,1])
            for i,a in enumerate(temp1):
                grupos[(temp1[i],temp2[i])] = k

        #aplica os grupos
        dados["grupo"] = dados.apply(lambda t:grupos[t["latlong"]],axis=1)
        agrupados = dados.groupby("grupo").sum()

        #checa se a diferença populacional entre os clusters é aceitável
        dif_max_min = np.max(agrupados["aptos_por_local"])-np.min(agrupados["aptos_por_local"])
        print(dif_max_min)
        #if dif_max_min > 4029:
        #    continue

        print("TAMANHO MEDIO DO CLUSTER: "+str(np.mean(agrupados["aptos_por_local"])))
        print("MAIOR CLUSTER: "+str(np.max(agrupados["aptos_por_local"])))
        print("MENOR CLUSTER: "+str(np.min(agrupados["aptos_por_local"])))
        print("DIFERENCA MAIOR E MENOR: "+str(np.max(agrupados["aptos_por_local"])-np.min(agrupados["aptos_por_local"])))

        for k in range(num_clusters):
            plot(data[idx==k,0],data[idx==k,1],'o')
        plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
        show()
        break

def converte_geojson():
    with open ("geojsonsampa.geojson") as f:
        data = json.load(f)
    saida = {"lat":[],"long":[]}
    for feature in data['features']:
        for property in feature["properties"]:
            if property not in saida:
                saida[property] = []
            saida[property].append(feature['properties'][property])
        saida["lat"].append(feature['geometry']['coordinates'][1])
        saida["long"].append(feature['geometry']['coordinates'][0])

    df = DataFrame(saida)
    del df["longitude"]
    del df["latitude"]
    del df["created_at"]
    del df["cartodb_id"]
    del df["updated_at"]

    df.to_csv("locais_com_votos.csv",index=False)

def arruma_votos():
    votos = read_csv("voto_secao_partido.csv")
    partidos = [13,45,15,55]
    votos = votos[votos.partido.isin(partidos)]
    votos["id"] = votos.apply(lambda t:str(t["num_zona"])+"--"+str(t["num_secao"]),axis=1)
    votos = DataFrame(votos.pivot_table(index=votos["id"], columns="partido",values="sum(votos)",aggfunc=np.sum))
    votos.columns = ["PT","PMDB","PSDB","PSD"]
    votos.to_csv("voto_secao_partido_trabalhada.csv")
    return votos

def junta_tabelas():
    #locais = quebra_secoes()
    votos = arruma_votos()
    print(votos)

    locais = read_csv("locais_com_votacao_trabalhada.csv")
    #votos = read_csv("voto_secao_partido_trabalhada.csv")
    saida = DataFrame.merge(locais,votos, left_on="id",right_on="id",how="outer")

    saida = DataFrame(saida.groupby(["lat","long","aptos_por_local","local_de_votacao","zona_eleitoral_nro","bairro","endereco","secoes_eleitorais","zona_eleitoral_nome"]).sum().reset_index())
    saida = saida.fillna(0)
    saida = saida[saida.secao != 0]
    saida["lat_real"] = saida["long"]
    saida["long"] = saida["lat"]
    saida["lat"] = saida["lat_real"]
    del saida["lat_real"]

    saida.to_csv("secoes_com_votacao.csv",index=False)


#converte_geojson()
#junta_tabelas()

plota_coordenadas()


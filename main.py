from pandas import read_csv, DataFrame
import numpy as np
import json
import random
from pylab import plot,show
from numpy import vstack,array
from scipy.cluster.vq import kmeans, kmeans2,vq
import math
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter

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

def cluster_algoritmo():
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


def eh_contiguo(a,b,distancia_padrao):
    return True if distancia_euclidiana(a,b) < distancia_padrao else False

def distancia_euclidiana(a,b):
    return math.sqrt(math.pow(a[0]-b[0],2)+math.pow(a[1]-b[1],2))*1000

def acha_proximo(a,lista,distancia_padrao,ja_tem_grupo):
    for item in lista:
        if item not in ja_tem_grupo:
            if eh_contiguo(a,item,distancia_padrao):
                return item
    #se acabar sem achar nenhum contiguo
    return False

def itera_grupo(grupo_atual,num_grupos,distancia):
    if grupo_atual < num_grupos:
        return grupo_atual+1,distancia
    else: return 1,distancia*1.05


def grupo_passou_tamanho(grupos, grupo, max_tamanho):
    itens_no_grupo = [i for i in grupos if grupos[i] == grupo]
    tamanho = sum([i[2] for i in itens_no_grupo])
    return tamanho > max_tamanho

def nao_terminou(grupos,itens):
    return len(grupos) == len(itens)

def invert_dict(d):
    newdict = {}
    for k, v in d.items():
        newdict.setdefault(v, []).append(k)
    return newdict

def acha_grupo(item,grupos,distancia_padrao):
    grupos = invert_dict(grupos)
    menor_distancia = distancia_padrao*2
    saida = False
    for grupo in grupos:
        distancia = distancia_euclidiana(item,grupos[grupo][0])
        if distancia < menor_distancia:
            menor_distancia = distancia
            saida = grupo

    return saida


def cluster_meu(num_grupos=55,max_iteracoes=10000,max_tamanho=155000,distancia_padrao=20):
    #arruma os dados
    dados = read_csv("secoes_com_votacao.csv")
    dados["latlongaptos"] = dados.apply(lambda t:(t["lat"],t["long"],t["aptos_por_local"]),axis=1)
    dados["latlong"] = dados.apply(lambda t:(t["lat"],t["long"]),axis=1)
    itens = list(dados["latlongaptos"])
    nao_tem_grupo = itens
    while True:
        grupos = {}
        iteracoes = 0
        grupo_atual = 1
        distancia_padrao=20
        ja_tem_grupo = []

        while nao_terminou(grupos,itens) or iteracoes < max_iteracoes:

            item = random.choice(itens)
            proximo = acha_proximo(item,nao_tem_grupo,distancia_padrao,ja_tem_grupo)

            if iteracoes % 100 == 0:
                print("ITERAÇÃO NÚMERO "+str(iteracoes))
                print("JÁ TEMOS "+str(len(ja_tem_grupo))+" ITENS COM GRUPO")


            #pula se não tiver achado proximo ou se o proximo por acaso for o mesmo que o item anterior
            if not proximo or proximo == item:
                continue

            #se o item já tiver um grupo, coloca o mesmo pro proximo
            if item in grupos:
                grupo = grupos[item]
                #mas só se o tamanho não chegou ainda no nosso médio
                if not grupo_passou_tamanho(grupos,grupo,max_tamanho):
                    grupos[proximo] = grupos[item]
                    ja_tem_grupo.append(proximo)
                    nao_tem_grupo.remove(proximo)

            #checa o mesmo para o proximo
            elif proximo in grupos:
                grupo = grupos[proximo]
                if not grupo_passou_tamanho(grupos,grupo,max_tamanho):
                    grupos[item] = grupos[proximo]
                    ja_tem_grupo.append(item)
                    nao_tem_grupo.remove(item)
            #se não, tenta achar um grupo próximo, no limite relativo à distância padrão atual
            else:
                grupo = acha_grupo(item,grupos,distancia_padrao)
                #se tiver esse grupo
                if grupo:
                    if not grupo_passou_tamanho(grupos,grupo,max_tamanho):
                        grupos[item] = grupo
                        grupos[proximo] = grupo
                        ja_tem_grupo += [item,proximo]
                        nao_tem_grupo.remove(item)
                        nao_tem_grupo.remove(proximo)
                #se não, cria um novo daí
                else:
                    grupo = grupo_atual
                    if not grupo_passou_tamanho(grupos,grupo_atual,max_tamanho):
                        grupos[item] = grupo
                        grupos[proximo] = grupo
                        ja_tem_grupo += [item,proximo]
                        nao_tem_grupo.remove(item)
                        nao_tem_grupo.remove(proximo)

            grupo_atual,distancia_padrao = itera_grupo(grupo_atual,num_grupos,distancia_padrao)
            iteracoes += 1

        print(distancia_padrao)
        print("ITERAÇÕES: "+str(iteracoes))
        print("FALTARAM "+str(len(itens)-len(grupos))+" LOCAIS DE VOTAÇÃO")

        #saida = invert_dict(grupos)

        dados["divisor"] = dados.apply(lambda t:grupos[t["latlongaptos"]] if t["latlongaptos"] in grupos else 0,axis=1)

        dados = dados[dados.divisor != 0]

        print("TEMOS "+str(len(set(dados["divisor"])))+" GRUPOS DIFERENTES")
        print("TAMANHO MAXIMO DE UM GRUPO: "+str(max([sum(dados[dados.divisor == z]["aptos_por_local"]) for z in set(dados["divisor"])])))
        print("TAMANHO MINIMO DE UM GRUPO: "+str(min([sum(dados[dados.divisor == z]["aptos_por_local"]) for z in set(dados["divisor"])])))
        print("TAMANHO MEDIO DE UM GRUPO: "+str(np.average([sum(dados[dados.divisor == z]["aptos_por_local"]) for z in set(dados["divisor"])])))

    cmap = plt.get_cmap('jet')
    colors = itertools.cycle([cmap(i) for i in np.linspace(0, 1, len(set(dados["divisor"])))])


    for z in set(dados["divisor"]):
        subdados = dados[dados["divisor"] == z]
        plt.scatter(subdados.lat,subdados.long, color=next(colors))
    plt.show()
    return

def distancia_media(itens):
    distancias = []
    for i in itens:
        for j in itens:
            if i != j:
                distancias.append(distancia_euclidiana(i,j))
    saida = np.average(distancias)
    print("DISTANCIA MEDIA: "+str(saida))
    return saida

def acha_seeds(itens,num_grupos):
    dados = [(i[0],i[1]) for i in itens]
    data = vstack(dados)
    centroids,_ = kmeans(data,num_grupos)
    return centroids


def eh_perto_de_seed(a,seeds,distancia):
    for s in seeds:
        if distancia_euclidiana(a,s) < distancia:
            return True
    return False

def acha_proximo_seed(seeds,itens,distancia):
    nao_seeds = [i for i in itens if i not in seeds]
    for n in nao_seeds:
        if not eh_perto_de_seed(n,seeds,distancia):
            return n,distancia
    return False,distancia*0.98

def distancia_seeds(item,seeds):
    saida = []
    for seed in seeds:
        saida.append(distancia_euclidiana(item,seed))
    return saida

def plota_grafico(dados):
    cmap = plt.get_cmap('jet')
    colors = itertools.cycle([cmap(i) for i in np.linspace(0, 1, len(set(dados["divisor"])))])

    for z in set(dados["divisor"]):
        subdados = dados[dados["divisor"] == z]
        plt.scatter(subdados.lat,subdados.long, color=next(colors))
    plt.show()

def acha_vencedor(grupo):

    return grupo

def calcula_resultado(dados):
    partidos = ["PMDB","PT","PSDB","PSD"]
    partidos.append("divisor")
    for coluna in dados.columns:
        if coluna not in partidos:
            del dados[coluna]

    grupos = dados.groupby(by=["divisor"]).sum()
    grupos["maior_num_votos"] = grupos.apply(max,axis=1)
    
    print(grupos)

#segunda tentativa, agora com um random seeding no começo
def cluster_meu2(num_grupos=55,max_iteracoes=10000,max_tamanho=155000,distancia_padrao=20):
    #arruma os dados
    dados = read_csv("secoes_com_votacao.csv")
    dados["latlongaptos"] = dados.apply(lambda t:(t["lat"],t["long"],t["aptos_por_local"]),axis=1)
    dados["latlong"] = dados.apply(lambda t:(t["lat"],t["long"]),axis=1)
    itens = list(dados["latlongaptos"])
    nao_tem_grupo = itens
    grupos = {}
    saida = {}

    seeds = acha_seeds(itens,num_grupos).tolist()
    seeds = [(i[0],i[1]) for i in seeds]
    dados["divisor"] = dados.apply(lambda t:grupos[t["latlongaptos"]] if t["latlongaptos"] in grupos else 0,axis=1)


    for item in itens:
        ordem_seeds = distancia_seeds(item,seeds)
        nao_colocou = True
        while nao_colocou:
            seed_atual = seeds[min(enumerate(ordem_seeds), key=itemgetter(1))[0]]
            if seed_atual not in grupos:
                grupos[seed_atual] = []
                grupos[seed_atual].append(item)
                saida[item] = seed_atual
                nao_colocou = False
            else:
                if not grupo_passou_tamanho(grupos,seed_atual,max_tamanho):
                    grupos[seed_atual].append(item)
                    saida[item] = seed_atual
                    nao_colocou= False
                else:
                    ordem_seeds.remove(seed_atual)
                    continue

    dados["divisor"] = dados.apply(lambda t:saida[t["latlongaptos"]] if t["latlongaptos"] in saida else 0,axis=1)
    print("TEMOS "+str(len(grupos))+ " GRUPOS")
    print("FALTARAM "+str(len(itens)-len(saida))+" LOCAIS DE VOTAÇÃO")

    resultado = calcula_resultado(dados)
    print(resultado)
    #plota_grafico(dados)





#converte_geojson()
#junta_tabelas()
#cluster_algoritmo()
#pl
cluster_meu2()


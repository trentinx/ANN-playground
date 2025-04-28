import random
import pickle
import numpy as np

class connection :
    def __init__(self,weight,neurin,neurout) :
        self.weight=weight
        self.neurin=neurin
        self.neurout=neurout
        self.valout=0
        neurin.connectout.append(self)
        neurout.connectin.append(self)
    
    def getout(self,val) :
        self.valout=val*self.weight
        self.neurout.signal(self.valout)
    
    def updt(self, delta, rate):
        # Calcul de l'update de poids
        nd = delta * self.neurin.sigy  # Calcul du gradient en fonction de l'erreur
        self.neurin.backprop(nd, self.weight, rate)  # Appliquer la rétropropagation à l'entrée
        self.weight = self.weight - rate * nd  # Mise à jour du poids

class neurone :
    def __init__(self,act,lamb,bias) :
        self.connectin=[]
        self.connectout=[]
        self.bias=bias
        self.lamb=lamb
        self.act=act
        self.sig=[]
        self.sigy=0
        self.nsig=0

    def inisignal(self, signal) :
        sign=self.act(signal,self.bias,self.lamb)
        for c in self.connectout :
            c.getout(sign)
        self.sigy=sign
    
    def signal(self,signal) :
        self.sig.append(signal)
        if len(self.sig)==len(self.connectin) :
            signal=0
            for i in self.sig :
                signal=signal+i
            self.sig=[]
            self.nsig=signal
            self.inisignal(signal)
    
    def inibackprop(self, err, rate):
        # Calcul de la dérivée de l'activation (fonction d'activation) pour obtenir la sensibilité du neurone
        De = self.act(self.nsig, self.bias, self.lamb, 1) * err  # Sensibilité de ce neurone

        # Mise à jour des poids de toutes les connexions entrantes
        for c in self.connectin:
            c.updt(De, rate)
    
        # Si c'est un neurone caché, propagons l'erreur vers la couche précédente
        if len(self.connectin) > 0:  # Vérifie qu'il y a des connexions entrantes (pour un neurone caché)
            self.sigy = De
    
    def backprop(self, delta, w, rate):
        self.sig.append(delta * w)
        # Si toutes les erreurs propagées sont reçues, calculez l'erreur totale
        if len(self.sig) == len(self.connectout):
            total_error = sum(self.sig)
            self.sig = []  # Réinitialisez les erreurs accumulées
            self.inibackprop(total_error, rate)

class layered_network :
    def __init__(self,nb_in,nb_out,nb_hidden,nb_perhidden,maxlamb,maxbias,maxstreigh,fonctex,foncthid) :
        self.layers=[]
        In=[]
        for i in range(nb_in) :
            In.append(neurone(fonctex,random.random()*maxlamb,random.random()*maxbias))
        self.layers.append(In)
        for i in range(nb_hidden) :
            L=[]
            for j in range(nb_perhidden) :
                L.append(neurone(foncthid,random.random()*maxlamb,random.random()*maxbias))
            self.layers.append(L)
        out=[]
        for i in range(nb_out) :
            out.append(neurone(fonctex,random.random()*maxlamb,random.random()*maxbias))
        self.layers.append(out)
        for i in range(len(self.layers)) :
            if not i==len(self.layers)-1 :
                L1=self.layers[i]
                L2=self.layers[i+1]
                for n1 in L1 :
                    for n2 in L2 :
                        C=connection(random.random()*maxstreigh,n1,n2)
        self.ready=0
    
    def getoutput(self,Input) :
        for n in range(len(Input)) :
            self.layers[0][n].inisignal(Input[n])
        out=[]
        for n in self.layers[len(self.layers)-1] :
            out.append(n.sigy)
        return out
    
    def onelearn(self, Input, expected, rate):
        # Obtenez la sortie du réseau
        output = self.getoutput(Input)
        # Calculez l'erreur pour chaque neurone de sortie
        dif = np.array(output)-np.array(expected)
        # Appliquez la rétropropagation pour chaque neurone de sortie
        for i, n in enumerate(self.layers[-1]):
            n.inibackprop(dif[i], rate)
    
    def multilearn(self, Input, expected, rate) :
        for i, I in enumerate(Input) :
            self.onelearn(I, expected[i], rate)
    
    def cyclelearn(self, Input,expected, rate,nb) :
        for i in range(nb) :
            self.multilearn(Input, expected, rate)

def relux(signal,bias,lamb,mod=0) :
    if mod==0 :
        a=signal+bias*lamb
        if a>0 :
            return a
        else :
            return 0
    if mod==1 :
        if signal+bias<=0 :
            return 0
        else :
            return lamb

def saveobject(Dir,obj) :
    with open ( Dir , "wb" ) as F :
        pickle.dump (obj , F )

def loadobject(Dir) :
    with open ( Dir , "rb" ) as F :
        return pickle.load ( F )

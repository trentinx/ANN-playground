import random
import numpy as np
import math
import joblib
import copy


class connection:
    def __init__(self, weight, neurin, neurout):
        self.weight = weight
        self.neurin = neurin  # neurone de départ
        self.neurout = neurout  # neurone d'arrivée
        self.valout = 0  # stocke la valeur de sortie
        # fait en sorte que les neurones contiennent leurs connexions
        neurin.connectout.append(self)
        neurout.connectin.append(self)
        
        self.delta=[]
        self.sumgrad=1

    def getout(self, val):
        self.valout = val * self.weight
        # Clipping de la valeur pour éviter des overflow
        self.valout = np.clip(self.valout, -1e6, 1e6)  # Limiter les sorties extrêmes
        self.neurout.signal(self.valout)

    def gradiant(self, Grad):
        self.delta.append(Grad*self.neurin.sigy)
        back=Grad*self.weight
        # Calcul de l'update de poids
        self.neurin.backprop(back)  # Propagation vers les entrées
    
    def updt(self, rate,lambd) :
        grad=sum(self.delta)/len(self.delta)
        self.sumgrad=(1-lambd)*self.sumgrad+lambd*grad**2
        nrate=rate/(math.sqrt(self.sumgrad)+10**(-8))
        
        self.weight=self.weight-grad*nrate
        self.delta=[]


class neurone:
    def __init__(self, act, lamb, bias):
        self.connectin = []  # connexions entrantes
        self.connectout = []  # connexions sortantes
        self.bias = bias
        self.lamb = lamb
        self.act = act  # fonction d'activation
        self.sig = []  # accumulation des signaux d'entrée
        self.sigy = 0
        self.nsig = 0
        self.deltaback = 0
        self.sigsave=[]
        self.gradsave=[]
        self.sumgrad=1

    def inisignal(self, signal):
        # Passe le signal dans la fonction d'activation
        sign = self.act(signal, self.bias, self.lamb)
        for c in self.connectout:
            c.getout(sign)
        self.sigy = sign

    def signal(self, signal):
        self.sig.append(signal)
        if len(self.sig) == len(self.connectin):  # Tous les signaux d'entrée sont reçus
            self.sigsave=copy.deepcopy(self.sig)
            total_signal = sum(self.sig)
            self.sig = []
            self.nsig = total_signal
            self.inisignal(total_signal)

    def inibackprop(self, err):
        Grad=self.act(self.nsig,self.bias,self.lamb,1)*err
        self.gradsave.append(Grad)
        for c in self.connectin :
            c.gradiant(Grad)

    def backprop(self, back):
        self.sig.append(back)
        if len(self.sig) == len(self.connectout):  # Tous les deltas sont reçus
            total_back = sum(self.sig)
            self.sig = []
            self.inibackprop(total_back)

    def updt(self,rate,lambd) :
        grad=sum(self.gradsave)/len(self.gradsave)
        self.sumgrad=(1-lambd)*self.sumgrad+lambd*grad**2
        nrate=rate/(math.sqrt(self.sumgrad)+10**(-8))
        
        self.bias=self.bias-grad*nrate
        self.gradsave=[]



class layered_network:
    def __init__(self, nb_in, nb_out, nb_hidden, nb_perhidden, maxlamb, maxbias, maxstreigh, fonctex, foncthid):
        self.layers = []
        # Initialisation des couches d'entrée
        In = [neurone(fonctex, random.uniform(0.1, maxlamb), random.uniform(0.1, maxbias)) for _ in range(nb_in)]
        self.layers.append(In)
        # Initialisation des couches cachées
        for _ in range(nb_hidden):
            hidden_layer = [neurone(foncthid, random.uniform(0.1, maxlamb), random.uniform(0.1, maxbias))
                            for _ in range(nb_perhidden)]
            self.layers.append(hidden_layer)
        # Initialisation des couches de sortie
        Out = [neurone(fonctex, random.uniform(0.1, maxlamb), random.uniform(0.1, maxbias)) for _ in range(nb_out)]
        self.layers.append(Out)
        # Création des connexions entre les couches
        for i in range(len(self.layers) - 1):
            L1, L2 = self.layers[i], self.layers[i + 1]
            for n1 in L1:
                for n2 in L2:
                    C = connection(random.uniform(-maxstreigh, maxstreigh), n1, n2)

    def getoutput(self, Input):
        for i, val in enumerate(Input):
            self.layers[0][i].inisignal(val)
        return [n.sigy for n in self.layers[-1]]

    def onelearn(self, Input, expected):
        output = self.getoutput(Input)
        dif = np.array(output) - np.array(expected)
        for i, n in enumerate(self.layers[-1]):
            n.inibackprop(dif[i])
        prim=lambda x:1/2*x**2
        Dif=np.array([prim(x) for x in dif])
        return sum(Dif)/len(Dif)

    def multilearn(self, Input, expected, rate,lambd):
        err=[]
        for i, I in enumerate(Input):
            err.append(self.onelearn(I, expected[i]))
        for L in self.layers :
            for n in L :
                n.updt(rate,lambd)
                for c in n.connectin :
                    c.updt(rate,lambd)
        return sum(err)/len(err)

    def cyclelearn(self, Input, expected, rate, nb,lambd):
        Err=[]
        for i in range(nb):
            Err.append(self.multilearn(Input, expected, rate,lambd))
        return Err

    def moyError(self, Input, expected):
        R = []
        for i in range(len(Input)):
            output = self.getoutput(Input[i])
            err = np.average(np.abs(np.array(output) - np.array(expected[i])))
            R.append(err)
        return np.average(R)

    def save(self, dir):
        saveobject(dir, self.layers)

    def load(self, dir):
        self.layers = loadobject(dir)


def sigmoid(signal, bias, lamb, mod=0):
    max_input = 500
    input_val = -(signal + bias) * lamb
    input_val = np.clip(input_val, -max_input, max_input)
    n = np.exp(input_val)
    if mod == 0:
        return 1 / (1 + n)
    elif mod == 1:
        sigmoid_val = 1 / (1 + n)
        return lamb * n * (sigmoid_val ** 2)


def tanh(signal, bias, lamb, mod=0):
    max_input = 100
    input_val = (signal + bias) * lamb
    input_val = np.clip(input_val, -max_input, max_input)
    if mod == 0:
        return np.tanh(input_val)
    elif mod == 1:
        return lamb * (1 - np.tanh(input_val) ** 2)


def relu(signal, bias, lamb, mod=0):
    if mod == 0:
        a = signal + bias * lamb
        return a if a > 0 else 0
    if mod == 1:
        return lamb if signal + bias > 0 else 0


def Lrelu(signal, bias, lamb, mod=0):
    if mod == 0:
        return lamb * (signal + bias) if signal + bias > 0 else (signal + bias) / lamb
    if mod == 1:
        return lamb if signal + bias > 0 else 1 / lamb


def saveobject(Dir, obj):
    with open(Dir, "wb") as F:
        joblib.dump(obj, F)


def loadobject(Dir):
    with open(Dir, "rb") as F:
        return joblib.load(F)


if __name__ == "__main__":
    inputs = []
    out = []
    num_test=4
    len_in=3
    len_out=4

    for i in range(num_test) :
        inp=[]
        for j in range(len_in) :
            inp.append(random.uniform(0,1))
        outp=[]
        for j in range(len_out) :
            val=0
            for i in range(len(inp)) :
                if i==j :
                    val=val+(i+1)*inp[i]
                else :
                    val=val+inp[i]**(i+1)
            if val>1 :
                val=val/10
            outp.append(val)
        inputs.append(inp)
        out.append(outp)

    L = layered_network(len_in, len_out, 5, 5, 1, 1, 1, sigmoid, Lrelu)
    print("Avant apprentissage :")
    for i, n in enumerate(inputs):
        print(f"Input: {n}, Output: {L.getoutput(n)}, Expected: {out[i]}")
        input()

    listerr=L.cyclelearn(inputs, out, 10**(-1), 1000,0.6)  # Taux d'apprentissage réduit et 5000 itérations

    print("\nAprès apprentissage :")
    for i, n in enumerate(inputs):
        print(f"Input: {n}, Output: {L.getoutput(n)}, Expected: {out[i]}")
        input()

    input()
    print(listerr)
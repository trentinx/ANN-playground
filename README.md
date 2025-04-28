# ANN-playground

## Pré-requis
Installer la lib ucimlrepo.
```bash
pip install ucimlrepo
```
Chargement des données et informations. 
```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
print(predict_students_dropout_and_academic_success.metadata) 
  
# variable information 
print(predict_students_dropout_and_academic_success.variables) 

```
## 1. Présentation du Perceptron Multicouches

### 1.1. Définition
Le Perceptron Multicouches (MLP) est une extension du perceptron à un seul neurone. Il s'agit d'un réseau de neurones
comportant plusieurs couches cachées entre la couche d'entrée et la couche de sortie, et qui permet de modéliser des
relations non linéaires complexes entre des features et une variable d'intérêt. Sa conception peut varier selon la tâche
à effectuer (classification ou régression), mais il se présente toujours sous une architecture classique, composée d'une
couche d'entrée, de couches cachées ou intermédiaires, et d'une couche de sortie.

### 1.2. Architecture classique

Le MLP se présente comme suit :

- une couche d'entrée : elle reçoit les caractéristiques d'entrées (attributs, features, variables explicatives), ce qui
permet de capturer l'information brute des données ; chacune de ses caractéristiques correspond à un neurone de la 
couche ; les caractéristiques peuvent être de plusieurs types : numériques, qualitatives, ou même des pixels d'une image
(qui sont en fait des valeurs numériques sous forme matricielle) ;
  
- une/plusieurs couches cachées : elles effectuent des transformations non linéaires et apprennent des représentations
intermédiaires des données ; chaque couche cachée est composée de neurones, et chaque neurone effectue deux opérations :
(a) calcul d'une somme pondérée des entrées provenant des neurones de la couche précédente ; (b) application d'une 
fonction d'activation pour introduire de la non-linéarité (ReLU, sigmoid ou tanh) ; le nombre et la taille des couches 
cachées déterminent la capacité du réseau à modéliser des relations complexes ;

- une couche de sortie : elle fournit la sortie finale du modèle, adaptée au problème à résoudre ; en classification, 
la couche de sortie contient souvent un neurone par classe, avec une fonction d'activation comme softmax (multiclasse)
ou sigmoid (binaire) ; en régression, elle contient généralement un seul neurone avec une activation linéaire.


### 1.3. Choix de l'architecture en fonction de la problématique de classification ou de régression

a- Classification

Couche d'entrée : le nombre de neurones correspond au nombre de caractéristiques d'entrée;

Couches cachées : une ou plusieurs couches cachées avec un nombre de neurones qui dépend de la complexité du problème ; 
le nombre de couches et de neurones peut être déterminé par des méthodes d'essai et d'erreur, la validation croisée, ou
des techniques de recherche hyperparamétrique ; plus de couches cachées et plus de neurones permettent de capturer des 
relations plus complexes ;

Couche de sortie : le nombre de neurones correspond au nombre de classes : pour une classification binaire, une seule
unité de sortie avec une fonction d'activation sigmoïde est utilisée, produit une probabilité entre 0 et 1 ; pour une
classification multiclasse, on utilise généralement une couche de sortie avec une fonction d'activation softmax, qui 
produit une distribution de probabilité sur les classes, ce qui est utile pour déterminer la classe la plus probable.


b- Régression

Couche d'entrée : le nombre de neurones correspond au nombre de caractéristiques d'entrée;

Couches cachées : une ou plusieurs couches cachées avec un nombre de neurones qui dépend de la complexité du problème.
Le nombre de couches et de neurones peut être déterminé par des méthodes d'essai et d'erreur, la validation croisée, 
ou des techniques de recherche hyperparamétrique ;

Couche de sortie : une seule unité de sortie avec une fonction d'activation linéaire (identité) ; Une seule unité de 
sortie avec une fonction d'activation linéaire produit une valeur continue, ce qui est idéal pour la régression.


## 2. Définition des termes relatifs au Perceptron Multicouches

### a. Fonction d'activation:
La fonction d'activation est une fonction mathématique, appliquée à la sortie (résultat) d'un neurone. Elle permet de 
décider si les neurones des couches intermédiaires et de sortie sont activés ou pas, en calculant la somme pondérée des 
valeurs en entrée, et en ajoutant un biais. Ceci permet de modéliser des relations complexes sur lesquelles se basent
des prédictions.

### b. Propagation :
C'est le processus de transmission des valeurs entrées, de la couche d'entrée vers la couche de sortie, en passant par 
les couches intermédiaires. Elle commence par une transformation opérée en faisant la somme pondérée des poids et des
valeurs d'entrée. Ensuite, cette transformation passe par une fonction d'activation, et donne en sortie la prédiction du
modèle.

### c. Retropropagation :
Ce processus permet au réseau de neurones d'apprendre en se basant sur les erreurs entre les valeurs prédites par le 
modèle en sortie et les valeurs réelles. Après le calcul de ces erreurs, le processus consiste à les propager vers 
l'arrière (i.e. vers les couches déjà parcourues par les données d'entrée), en ajustant les poids et le biais. Pour cela, 
on utilise souvent l'algorithme de la descente du gradient, qui consiste à calculer le gradient afin de naviguer
efficacement dans les couches complexes du réseau neuronal, et de minimiser la fonction de coût.


### d. Loss function : 
Une fonction mathématique qui mesure le degré de correspondance entre les prédictions du modèle et les valeurs réelles. 
Elle fournit une mesure quantitative de la précision des prédictions du modèle, qui peut être utilisée pour guider le 
processus d'amélioration du modèle. L'objectif est de guider les algorithmes d'optimisation dans l'ajustement des 
paramètres du modèle afin de réduire cette perte.

### e. Descente de gradients : 
La descente de gradients est un algorithme qui consiste à ajuster les paramètres du modèle en petites étapes dans la direction opposée du gradient, permettant ainsi de réduire progressivement la loss function. L'lgorithme se présente comme suit:

    - Initialisation : on initialise les paramètres (poids) du modèle avec des valeurs aléatoires.
    - Itération : pour chaque itération (épisode d'entraînement) :
        --Calculer le gradient de la fonction de coût par rapport aux paramètres.
        --Mettre à jour les paramètres en soustrayant un multiple du gradient (appelé taux d'apprentissage) des paramètres actuels.
    -Convergence : l'entraînement continue jusqu'à ce que la fonction de coût soit minimisée ou que le changement de la fonction de coût devienne négligeable.


### f. Vanishing gradients

Ce phénomène se manifeste lorsque les gradients deviennent de plus en plus petits au fur et à mesure que l'on remonte les couches du réseau lors de la rétropropagation. Cela peut rendre l'entraînement très lent ou même impossible pour certaines couches (saturation des neurones). Pour résoudre ce problème, on peut:
    - utiliser des fonctions d'activation qui atténuent le problème, comme ReLU (Rectified Linear Unit) au lieu de sigmoïde ou tangente hyperbolique;
    - utiliser des méthodes d'initialisation des poids appropriées, comme l'initialisation de He ou de Xavier;
    - utiliser des réseaux résiduels (ResNets), qui établissent des connexions résiduelles pour permettre aux gradients de se propager plus facilement à travers les couches;
    - enfin, utiliser des optimiseurs avancés comme Adam ou RMSprop qui peuvent aider à atténuer le problème


## 3. Présentation d'hyperparamètres

 a. Taux d'apprentissage: détermine la taille des pas que l'algorithme de descente de gradients prend pour ajuster les poids du réseau:

  - initialisation: commencer avec une valeur moyenne, généralement entre 0.001 et 0.1;
  - réduction progressive: utiliser une réduction progressive du taux d'apprentissage (learning rate decay) pour permettre une convergence plus fine;
  - recherche par grille ou aléatoire: effectuer un grid search pour trouver la meilleure valeur;
  - recherche automatisée: utiliser des méthodes d'optimisation automatisées comme Optuna ou Hyperopt.

 b. Nombre de couches cachées: détermine la profondeur du réseau:

  - problèmes simples: une ou deux couches cachées peuvent suffire;
  - problèmes complexes: pour des problèmes complexes, comme la reconnaissance d'images ou le traitement du langage naturel, plusieurs couches (10-100) peuvent être nécessaires;
  - éviter la surcomplexité: commencer avec un nombre de couches modéré et augmenter progressivement si nécessaire pour éviter la surcomplexité et l'overfitting.

 c. Nombre de neurones par couche: détermine la capacité du réseau à apprendre des représentations complexes:
 
  - taille modérée: commencer avec un nombre de neurones modéré et ajuster en fonction des résultats;
  - réduction progressive du nombre de neurones à mesure que l'on avance dans le réseau;
  - éviter l'underfitting et l'overfitting** : augmenter le nombre de neurones si le modèle underfit, et le réduire si le modèle overfit.

 d. Régularisation, utilisée pour prévenir l'overfitting en ajoutant une pénalité aux poids du modèle:
 
  - L1 et L2: utiliser L1 (Lasso) pour favoriser la sparsité des poids, ou L2 (Ridge) pour réduire la magnitude des poids;
  - dropout: utiliser le dropout pour désactiver aléatoirement des neurones pendant l'entraînement, ce qui aide à prévenir l'overfitting;
  - early stopping: arrêter l'entraînement lorsque la performance sur un ensemble de validation commence à diminuer;
  - hyperparamètres: ajuster les coefficients de régularisation (lambda) en fonction des résultats, généralement entre 0.0001 et 0.1.

 e. Taille du lot (batch size),détermine le nombre d'exemples d'entraînement traités avant chaque mise à jour des poids du réseau

  - synchroniser avec le taux d'apprentissage pour garder un nombre d'exemples équivalent par mise à jour;
  - surveiller la généralisation en mesurant l'erreur sur chaque jeu de validation

 

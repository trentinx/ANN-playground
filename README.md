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
Le Perceptron Multicouches (MLP) est une extension du perceptron à un seul neurone. Il s'agit d'un réseau de neurones comportant plusieurs couches cachées entre la couche d'entrée et la couche de sortie, et qui permet de modéliser des relations non linéaires complexes entre des features et une variable d'intérêt.
Sa conception peut varier selon la tâche à effectuer (classification ou régression), mais il se présente toujours sous une architecture classique, composée d'une couches d'entrée, de couches cachées ou couches intermédiaires, et d'une couche de sortie.

### 1.2. Architecture classique

Le MLP se présente comme suit:

- une couche d'entrée: elle reçoit les caractéristiques d'entrées (attributs, features, variables explicatives), ce qui permet de capturer l'information brute des données; chacune de ses caractéristiques correspond à un neurone de la couche; les caractéristiques peuvent être de plusieurs types: numériques, qualitatives, ou même des pixels d'une image (qui sont en fait des valeurs numériques sous forme matricielle);
  
- une/plusieurs couches cachées: elles effectuent des transformations non linéaires et apprennent des représentations intermediaires des données; chaque couche cachée est composée de neurones, et chaque neurone effectue deux opérations: (a) calcul d'une somme pondérée des entrées provenant des neurones de la couche précédente; (b) application d'une fonction d'activation pour introduire de la non-linéarité (ReLU, sigmoid ou tanh); le nombre et la taille des couches cachées déterminent la capacité du réseau à modéliser des relations complexes;

- une couche de sortie: elle fournit la sortie finale du modèle, adaptée au problème à résoudre; en classification, la couche de sortie contient souvent un neurone par classe, avec une fonction d'activation comme softmax (multiclasse) ou sigmoid (binaire); en régression, elle contient généralement un seul neurone avec une activation linéaire.


### 1.3. Choix de l'architecture en fonction de la problématique de classification ou de régression

a- Classification

Couche d'entrée: le nombre de neurones correspond au nombre de caractéristiques d'entrée;

Couches cachées: une ou plusieurs couches cachées avec un nombre de neurones qui dépend de la complexité du problème; le nombre de couches et de neurones peut être déterminé par des méthodes d'essai et d'erreur, la validation croisée, ou des techniques de recherche hyperparamétrique; plus de couches cachées et plus de neurones permettent de capturer des patterns plus complexes;

Couche de sortie: le nombre de neurones correspond au nombre de classes: pour une classification binaire, une seule unité de sortie avec une fonction d'activation sigmoïde est utilisée, produit une probabilité entre 0 et 1; pour une classification multiclasse, on utilise généralement une couche de sortie avec une fonction d'activation softmax, qui produit une distribution de probabilité sur les classes, ce qui est utile pour déterminer la classe la plus probable.


b- Régression

Couche d'entrée: le nombre de neurones correspond au nombre de caractéristiques d'entrée;

Couches cachées: une ou plusieurs couches cachées avec un nombre de neurones qui dépend de la complexité du problème. Le nombre de couches et de neurones peut être déterminé par des méthodes d'essai et d'erreur, la validation croisée, ou des techniques de recherche hyperparamétrique;

Couche de sortie: une seule unité de sortie avec une fonction d'activation linéaire (identité); Une seule unité de sortie avec une fonction d'activation linéaire produit une valeur continue, ce qui est idéal pour la régression.



## 2. Définition des termes relatifs au Perceptron Multicouches



## 3. Présentation d'hyperparamètres


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tree import EvoParams, EvolutionaryForest
from utils import prepare_data


def main():
    
    train = pd.read_csv('datasets/space_titanic/train.csv')
    data = prepare_data(train, 'Transported', drop_columns=['PassengerId', 'Name', 'Cabin'])
    
    n_classifiers = 10
    
    params = EvoParams(
        population_size=1000,
        crossover_rate=0.7,
        mutation_rate=0.4,
        n_elites=50,
        n_classifiers=n_classifiers,
        early_stopping_patiance=20,
        training_sample_size = data['X_train'].shape[0] // 2,
        initial_temperature=1,
        n_species = 10
    )
    
    classifier_sizes = []
    evo_n_nodes = []
    evo_accuracies = []
    
    sk_learn_sizes = []
    sk_learn_node_sizes = []
    sk_learn_accuracies = []
    
    for i in range(10):
        evo = EvolutionaryForest(data, params)
        evo.evolve(100)
    
        
        classifiers = [5, 10, 25, 50, 75, 100]
        for n in classifiers:
            evo.build_ensemble(n)
            predictions = evo.predict(data['X_test'])
            accuracy = accuracy_score(data['y_test'], predictions)
            nodes_sizes = [len(tree.values) for tree in evo.ensemble]
            classifier_sizes.append(n)
            evo_n_nodes.append(nodes_sizes)
            evo_accuracies.append(accuracy)
            print(f"Evolutionary {n} classifiers accuracy: {accuracy}")
            
            sk_learn = RandomForestClassifier(n_estimators=n)
            sk_learn.fit(data['X_train'], data['y_train'])
            sk_learn_accuracy = sk_learn.score(data['X_test'], data['y_test'])
            sk_nodes_sizes = [tree.tree_.node_count for tree in sk_learn.estimators_]
            sk_learn_sizes.append(n)
            sk_learn_node_sizes.append(sk_nodes_sizes)
            sk_learn_accuracies.append(sk_learn_accuracy)
            print(f"Sklearn Forest {n} classifiers accuracy: {sk_learn_accuracy}")
        
        """ file = open(f"results/trees_{n}_classifiers.txt", 'w')
        file.write(f"Evolutionary Forest {n} classifiers\n")
        file.write(params.__str__())
        file.write('\n'.join(str(tree) for tree in evo.ensemble)) """
        
    df1 = pd.DataFrame({
        'n_classifiers': classifier_sizes,
        'n_nodes': evo_n_nodes,
        'accuracy': evo_accuracies
    })
    
    df2 = pd.DataFrame({
        'n_classifiers': sk_learn_sizes,
        'n_nodes': sk_learn_node_sizes,
        'accuracy': sk_learn_accuracies
    })
    
    df1.to_csv('results/spaceship_evolutionary_forest.csv', index=False)
    df2.to_csv('results/spaceship_sklearn_forest.csv', index=False)
        
if __name__ == '__main__':
    main()
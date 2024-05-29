import ast
import os
from matplotlib import pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns



def scatter_plot(path1, path2, name, save_folder='results'):
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    
    df1['Source'] = 'Evolutionary Forest'
    df2['Source'] = 'Scikit-learn Forest'
    
    df = pd.concat([df1, df2])
    df['n_nodes'] = df['n_nodes'].apply(ast.literal_eval)
    df['avg_nodes'] = df['n_nodes'].apply(lambda x: sum(x))
    
    plt.figure(figsize=(10, 5.7097))
    
    sns.scatterplot(x='avg_nodes', y='accuracy', hue='Source', data=df, alpha=0.7)
    
    plt.xlabel('Sum of Nodes in Forest')
    plt.ylabel('Prediction Accuracy')
    plt.legend()
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    plt.savefig(f'{save_folder}/{name}_scatter.png', dpi=300, bbox_inches='tight')

def barplot_results(path1, path2, name, save_folder='results'):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df1['Source'] = 'Evolutionary Forest'
    df2['Source'] = 'Scikit-learn Forest'

    df = pd.concat([df1, df2])

    plt.figure(figsize=(10, 5.7097))
    sns.barplot(x='n_classifiers', y='accuracy', hue='Source', data=df)
    plt.xlabel('Forest Size')
    plt.ylabel('Prediction Accuracy')
    plt.grid(True)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    plt.savefig(f'{save_folder}/{name}.png', dpi=300, bbox_inches='tight')
    
    
def plot_results(path, name, save_folder='results'):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    df = pd.read_csv(path)
    df['n_nodes'] = df['n_nodes'].apply(ast.literal_eval)
    
    plt.figure(figsize=(10, 5.7097))
    sns.boxplot(x='n_classifiers', y='accuracy', data=df)
    plt.xlabel('Forest Size')
    plt.ylabel('Prediction Accuracy')
    plt.grid(True)
    plt.savefig(f'{save_folder}/{name}_accuracy_vs_classifiers.png', dpi=300, bbox_inches='tight')


def plot_combined_boxplots(path1, path2, name, save_folder='results'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df1['Source'] = 'Evolutionary Forest'
    df2['Source'] = 'Scikit-learn Forest'

    df = pd.concat([df1, df2])
    df['n_nodes'] = df['n_nodes'].apply(ast.literal_eval)

    plt.figure(figsize=(10, 5.7097))
    sns.boxplot(x='n_classifiers', y='accuracy', hue='Source', data=df)
    plt.xlabel('Forest Size')
    plt.ylabel('Prediction Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{save_folder}/{name}_combined_boxplot.png', dpi=300, bbox_inches='tight')


def main():
    
    params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
    rc(params) 

    
    scatter_plot(r'results\darwin_evolutionary_forest.csv', r'results\darwin_sklearn_forest.csv', 'darwin_scatter', 'plots/darwin')
    scatter_plot(r'results\spaceship_evolutionary_forest.csv', r'results\spaceship_sklearn_forest.csv', 'spaceship_scatter', 'plots/spaceship')
    scatter_plot(r'results\toxicity_evolutionary_forest.csv', r'results\toxicity_sklearn_forest.csv', 'toxicity_scatter', 'plots/toxicity')
    scatter_plot(r'results\Diabetes_binary_evolutionary_forest.csv', r'results\Diabetes_binary_sklearn_forest.csv', 'diabetes_scatter', 'plots/diabetes')
    
    barplot_results(r'results\darwin_evolutionary_forest.csv', r'results\darwin_sklearn_forest.csv', 'darwin_barplot', 'plots')
    barplot_results(r'results\spaceship_evolutionary_forest.csv', r'results\spaceship_sklearn_forest.csv', 'spaceship_barplot', 'plots')
    barplot_results(r'results\toxicity_evolutionary_forest.csv', r'results\toxicity_sklearn_forest.csv', 'toxicity_barplot', 'plots')
    barplot_results(r'results\Diabetes_binary_evolutionary_forest.csv', r'results\Diabetes_binary_sklearn_forest.csv', 'diabetes_barplot', 'plots')
    
    plot_combined_boxplots(r'results\darwin_evolutionary_forest.csv', r'results\darwin_sklearn_forest.csv', 'darwin', 'plots/darwin')
    plot_combined_boxplots(r'results\spaceship_evolutionary_forest.csv', r'results\spaceship_sklearn_forest.csv', 'spaceship', 'plots/spaceship')
    plot_combined_boxplots(r'results\toxicity_evolutionary_forest.csv', r'results\toxicity_sklearn_forest.csv', 'toxicity', 'plots/toxicity')
    plot_combined_boxplots(r'results\Diabetes_binary_evolutionary_forest.csv', r'results\Diabetes_binary_sklearn_forest.csv', 'diabetes', 'plots/diabetes')

    
if __name__ == '__main__':
    main()

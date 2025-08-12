#! /usr/bin/env python3

#from pytetrad.tools import TetradSearch

import json
import os
from pathlib import Path
import socket
from typing import Optional, List

from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler

# set the PATH if needed
hostname = socket.gethostname() 
home_directory = Path.home()
# check if .javaenv.txt file exists
javaenv_path = os.path.join(home_directory, '.javarc')
if os.path.exists(javaenv_path):
    # load the file
    load_dotenv(dotenv_path=javaenv_path)
    java_home = os.environ.get("JAVA_HOME")
    java_path = f"{java_home}/bin"
    current_path = os.environ.get('PATH')
    # add this to PATH
    os.environ['PATH'] = f"{current_path}{os.pathsep}{java_path}"

    # add to path
    graphviz_bin = os.environ.get("GRAPHVIZ_BIN")
    os.environ['PATH'] = f"{current_path}{os.pathsep}{graphviz_bin}"

    pass

elif 'c30' in hostname:
    java_home = "R:/DVBIC/jdk21.0.4_7"
    # init JAVA_HOME
    os.environ['JAVA_HOME'] = java_home
    java_path = f"{java_home}/bin"
    current_path = os.environ.get('PATH')
    # add this to PATH
    os.environ['PATH'] = f"{current_path}{os.pathsep}{java_path}"
    pass

import graphviz
import re
import pytetrad.tools.translate as tr
import pandas as pd
import semopy
import tqdm

__version_info__ = ('0', '2', '10')
__version__ = '.'.join(__version_info__)

version_history = \
"""
0.2.10 - add getTetradVersion to get the version of Tetrad
0.2.9 - add option to run_stability_search save_file to save the results to a json file
0.2.8 - add intermediate results for get_hyper_parameters providing results of
        each run
0.2.7 - add get_hyper_parameters to calculate the hyper parameters to achieve a target 
        false positive rate for a specific dataset
0.2.6 - set default verbose to False to stop the adjacentNodes messages
0.2.5 - added progress bar to run_stability_search
0.2.4 - added stability analysis with edge merging
0.2.3 - add generate_permuted_dfs - permutes data in each column
0.2.2 - generate semopy plot
0.2.1 - add subsample_df and reading in csv
0.2.0 - add reading of .javarc for JAVA_HOME
0.1.2 - reworked data file paths
0.1.1 - refactored to generalize operations with run_model_search
0.1.0 - initial version  
"""

# Correctly import the CLASS 'TetradSearch' from WITHIN the MODULE 'TetradSearch'
try:
    from pytetrad.tools.TetradSearch import TetradSearch as TetradSearchBaseClass
    #print(f"DEBUG: Successfully imported TetradSearchBaseClass, type: {type(TetradSearchBaseClass)}")
    # Optional check to be absolutely sure it's a class now
    if not isinstance(TetradSearchBaseClass, type):
        print("ERROR: Imported object is not actually a class type!")
        exit()
except ImportError:
    print("FATAL ERROR: Could not import the 'TetradSearch' class from the 'pytetrad.tools.TetradSearch' module.")
    print("Please double-check the library's structure and your installation.")
    # You might want to explore the structure if the import fails:
    # import pytetrad.tools
    # print("Contents of 'pytetrad.tools':", dir(pytetrad.tools))
    # import pytetrad.tools.TetradSearch
    # print("Contents of 'pytetrad.tools.TetradSearch':", dir(pytetrad.tools.TetradSearch))
    exit()
except Exception as e:
    print(f"FATAL: An unexpected error occurred during import: {e}")
    exit()
#print(f"DEBUG: Type of imported TetradSearch: {type(TetradSearchBaseClass)}")
# Add this to see its attributes, might help identify if it's a class or module
# print(f"Attributes of TetradSearch: {dir(TetradSearch)}")

from edu.cmu.tetrad.util import Params, Parameters

class MyTetradSearch(TetradSearchBaseClass):
    def __init__(self):

        # create a dummy dataframe to init object
        dummy_df = pd.DataFrame({
            'dummy1': [1, 2, 3, 4, 5],
            'dummy2': [5, 4, 3, 2, 1]
        })
        # Call the parent class's constructor
        # super().__init__(*args, **kwargs)
        super().__init__(dummy_df)
        
    def getTetradVersion(self):
        """
        Get the version of Tetrad
        """
        
        from edu.cmu.tetrad.util import Version
        
        version = Version.currentViewableVersion().toString()
        return version

    def read_csv(self, file_path: str) -> pd.DataFrame:
        """Read a CSV file and return a pandas DataFrame.
        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: pandas DataFrame
        """
        self.full_df = pd.read_csv(file_path)
        return self.full_df

    def add_lag_columns(self, df: pd.DataFrame, lag_stub='_') -> pd.DataFrame:
        """
        Lag the dataframe by shifting the columns by one row

        Args:
            df (pd.DataFrame): the dataframe to lag
            lag_stub (str): the string to append to the column names for the lagged variables

        Returns:
            pd.DataFrame: the lagged dataframe
        """
        
        # create a copy of the dataframe
        df_lag = df.copy()
        
        # create additional columns for the lagged variables, naming them  lagdrinks, lagsad, etc.
        cols_to_lag = df.columns.tolist()
        # shift by one row
        for col in cols_to_lag:
            df_lag[f'{col}{lag_stub}'] = df[col].shift(1)
        
        # drop the first row
        df_lag = df_lag.dropna()
        
        # reset index
        df_lag = df_lag.reset_index(drop=True)
        
        return df_lag
    
    def load_df_into_ts(self,df):
        """
        Loads a pandas DataFrame into the TetradSearch object.
        """
        self.data = tr.pandas_data_to_tetrad(df)
        return self.data
    
    def subsample_df(self, df: Optional[pd.DataFrame] = None,    
                    fraction: float = 0.9,
                    random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Randomly subsample the DataFrame to a fraction of rows
        Args:
            df - pandas DataFrame
            fraction - proportion of rows to keep, default 0.9
            random_state - random state for reproducibility
        Returns:
            df - pandas DataFrame
        """
        if df is None:
            if hasattr(self, 'full_df') and self.full_df is not None:
                df = self.full_df
            else:
                raise ValueError("DataFrame must be provided.")
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"fraction must be between 0 and 1")
        
        # Use the DataFrame's built-in sample method with the specified fraction
        # The `random_state` parameter ensures reproducibility if an integer is provided
        scrambled_df = df.sample(frac=fraction, random_state=random_state)

        # Step 2: Sort the sampled DataFrame by its original index
        # This restores the original relative order of the kept rows
        self.subsampled_df = scrambled_df.sort_index()
        
        return self.subsampled_df

    def standardize_df_cols(self, df, diag=False):
        """
        standardize the columns in the dataframe
        https://machinelearningmastery.com/normalize-standardize-machine-learning-data-weka/
        
        * get the column names for the dataframe
        * convert the dataframe into  a numeric array
        * scale the data
        * convert array back to a df
        * add back the column names
        * set to the previous df
        """
        
        # describe original data - first two columns
        if diag:
            print(df.iloc[:,0:2].describe())
        # get column names
        colnames = df.columns
        # convert dataframe to array
        data = df.values
        # standardize the data
        std_data = StandardScaler().fit_transform(data)
        # convert array back to df, use original colnames
        newdf = pd.DataFrame(std_data, columns = colnames)
        # describe new data - first two columns
        if diag:
            print(newdf.iloc[:,0:2].describe())
        
        return newdf
    
    def create_permuted_dfs(self, df: pd.DataFrame, n_permutations: int, seed: int = None) -> List[pd.DataFrame]:
        """
        Generates multiple DataFrames, each with elements permutated within columns independently.

        Args:
            df: The input pandas DataFrame.
            n_permutations: The number of permutated DataFrames to generate.
            seed: An optional integer seed for the random number generator
                to ensure reproducibility of the sequence of permutations.
                If None, the permutations will be different each time.

        Returns:
            A list containing n pandas DataFrames, each being a column-wise
            permutation of the original DataFrame.
        """
        # Check if the input DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty. Please provide a valid DataFrame.")

        # Check if n is a positive integer
        if not isinstance(n_permutations, int) or n_permutations <= 0:
            raise ValueError("The number of permutations (n) must be a positive integer.")

        # If a seed is provided, set it for reproducibility
        # This allows for consistent random permutations across different runs
        # If no seed is provided, the permutations will be different each time
        # Note: np.random.default_rng() is used to create a new random number generator instance
        # This is a more modern approach compared to np.random.seed() and allows for better control
        # over random number generation.
        # The seed is used to initialize the random number generator
        # This ensures that the same sequence of random numbers is generated each time
        # the same seed is used.
        # This is useful for debugging and testing purposes
        # If no seed is provided, the permutations will be different each time
        # the function is called.   
        # Initialize the random number generator once
        # If a seed is provided, the sequence of generated permutations will be reproducible
        rng = np.random.default_rng(seed)

        permutated_dfs = [] # List to store the resulting DataFrames

        for _ in range(n_permutations): # Generate n permutations
            # Create a fresh copy of the original DataFrame for each permutation
            df_permuted = df.copy()

            # Permutate each column independently using the same RNG state
            for col in df_permuted.columns:
                df_permuted[col] = rng.permutation(df_permuted[col].values)

            # Add the newly permutated DataFrame to the list
            permutated_dfs.append(df_permuted)

        return permutated_dfs

    def run_stability_search(self, full_df: pd.DataFrame,
                            model: str = 'gfci',
                            knowledge: Optional[dict] = None,
                            score = {'sem_bic': {'penalty_discount': 1.0}},
                            test ={'fisher_z': {'alpha': .05}},
                            runs: int = 100,
                            min_fraction: float= 0.75,
                            subsample_fraction: float = 0.9,
                            random_state: Optional[int] = None,
                            lag_flag = False,
                            save_file: Optional[str] = None) -> tuple:
        """
        Run a stability search on the DataFrame using the specified model.
        
        Edges are tabluated for each run using a set.
        Edges that are present for a minimum of min_fraction of runs will be retained
        and returned.
        The edges are returned as a list of strings.
        
        Args:
            df: pd.DataFrame - the DataFrame to perform the stability search on
            model: str - the model to use for the search
            knowledge: Optional[dict] - additional knowledge to inform the search
            score: dict - scoring parameters
            test: dict - testing parameters
            runs: int - number of runs to perform
            min_fraction: float - minimum fraction of runs an edge must appear in to be retained
            subsample_fraction: float - fraction of data to subsample for each run
            random_state: Optional[int] - random state for reproducibility
            lag_flag: - if True, add lagged columns to the DataFrame
        Returns:
            list - list with the results
        """

        # dictionary where key is the edge and value is the number of times it was found
        edge_counts = {}
        
        # list to hold results of each run
        run_results = []

        # check if in jupyter to select the progress bar code
        if self.in_jupyter():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm      
              
        # loop over the number of runs
        myRuns = range(runs)
        for i in tqdm(myRuns, desc=f"Running stability search with {runs} runs", unit="run"):
            # subsample the DataFrame
            df = self.subsample_df(full_df, fraction=subsample_fraction, random_state=random_state)
            
            # check if lag_flag is True
            if lag_flag:
                # add lagged columns
                df = self.add_lag_columns(df, lag_stub='_lag')
                
            # standardize the data
            df = self.standardize_df_cols(df)
                
            # run the search
            searchResult = self.run_model_search(df, model=model, 
                                                knowledge=knowledge, 
                                                score=score,
                                                test=test,
                                                verbose=True)
            # get the edges
            edges = searchResult['setEdges']
            # loop over the edges
            for edge in edges:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

            info = {
                'edges': edges,
                'edge_counts': edge_counts,
            }
            # add the info to the runs list
            run_results.append(info)
        print(f"\nSearch complete!")
        
        # check similarity of edges, sort alphabetically
        # get all the keys and then sort them

        sorted_edge_keys = sorted(edge_counts.keys())
        
        sorted_edge_counts = {}
        sorted_edge_counts_raw = {}
        # loop over the sorted keys and store the fractional count 
        for edge in sorted_edge_keys:
            sorted_edge_counts[edge] = edge_counts[edge]/runs
            sorted_edge_counts_raw[edge] = edge_counts[edge]

        selected_edges = self.select_edges(sorted_edge_counts, min_fraction=min_fraction)

        # combine results into a dictionary
        results = {
            'selected_edges': selected_edges,
            'sorted_edge_counts': sorted_edge_counts,
            'sorted_edge_counts_raw': sorted_edge_counts_raw,
            'edge_counts': edge_counts, 
            #'run_results': run_results, # error is not JSON serializable
        }

        if save_file is not None:
            # save the results to a json file
            with open(save_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {save_file}")

        return selected_edges, sorted_edge_counts, sorted_edge_counts_raw, run_results
    
    def select_edges(self, edge_counts: dict, min_fraction: float) -> dict:
        """
        Select edges based on a fraction of the total edges
        Args:
            edges - dictionary with the edges and their counts
            min_fraction - mininum fraction of edges to select
        Returns:
            dict - dict of selected edges as keys and their fraction as values
        """

        selected_edges = {}  # holds the selected edges as strings e.g. 'A --> B' with fraction
        
        # create a dataframe with src, edge, dest and fraction, extract the src, edge, dest from each edge
        # and add to the dataframe
        edge_df = pd.DataFrame(edge_counts.items(), columns=['edge', 'fraction'])
        edge_df[['src', 'edge_type', 'dest']] = edge_df['edge'].str.split(' ', expand=True)
        # drop the edge column
        edge_df = edge_df.drop(columns=['edge'])
        
        # first lets process the edge_type of --> and o->
        # filter the df for edge_type of --> and o->
        directed_edge_df = edge_df[edge_df['edge_type'].isin(['-->', 'o->'])]
        # sort the directed edges by src, dest, and fraction (descending)
        directed_edge_df = directed_edge_df.sort_values(by=['src', 'dest', 'fraction'], ascending=[True, True, False])

        # get the rows where the src and dest only appear once in the df
        single_directed_edges = directed_edge_df.groupby(['src', 'dest']).filter(lambda x: len(x) == 1)
        
        # iterate over the rows and add them to the edge list if the fraction is >= min_fraction
        for index, row in single_directed_edges.iterrows():
            if row['fraction'] >= min_fraction:
                # create the edge string
                edge_str = f"{row['src']} {row['edge_type']} {row['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = row['fraction']
                pass
            
        # now lets process the directed edges that have multiple rows                   
        # keep the rows where the src and dest are the same across rows
        multiple_directed_edges = directed_edge_df.groupby(['src', 'dest']).filter(lambda x: len(x) > 1)
        
        # iterate over two rows, and if the sum of fraction is greater than min_fraction then keep the edge
        for i in range(0, len(multiple_directed_edges), 2):
            row_pairs = multiple_directed_edges.iloc[i:i+2]
            # get the sum of the fraction
            fraction_sum = row_pairs['fraction'].sum()
            # check if the sum is greater than min_fraction
            if fraction_sum >= min_fraction:
                # create the edge string
                # we use the first row to create the edge string since it has the highest fraction
                edge_str = f"{row_pairs.iloc[0]['src']} {row_pairs.iloc[0]['edge_type']} {row_pairs.iloc[0]['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = float(fraction_sum)
                pass
            pass
        
        # get the undirected edges
        undirected_edges_df = edge_df[~edge_df['edge_type'].isin(['-->', 'o->'])]

        # iterate over the df using iloc, if the src > dest then swap them in the row and update the df
        # this will make sure that the edges are in the same order
        # for example, A o-o B and B <-> A will be made adjacent to each other when sorted
        for i in undirected_edges_df.index:
            # Use .loc to access the specific row and columns directly
            # This ensures you are working with the original DataFrame
            # and prevents the SettingWithCopyWarning
            if undirected_edges_df.loc[i, 'src'] > undirected_edges_df.loc[i, 'dest']:
                # Swap the values directly in the DataFrame using .loc
                # We use tuple assignment for a clean swap
                undirected_edges_df.loc[i, 'src'], undirected_edges_df.loc[i, 'dest'] = \
                undirected_edges_df.loc[i, 'dest'], undirected_edges_df.loc[i, 'src']
            
            
            # row = undirected_edges_df.iloc[i]
            # if row['src'] > row['dest']:
            #     # swap the src and dest
            #     temp = row['src']
            #     row['src'] = row['dest']
            #     row['dest'] = temp
            #     # update the row in the df using iloc
            #     undirected_edges_df.iloc[i] = row
            pass
    
        # sort the undirected edges by src, dest, and fraction (descending)
        undirected_edges_df = undirected_edges_df.sort_values(by=['src', 'dest', 'fraction'], ascending=[True, True, False])
        # get the undirected edges that have single rows
        single_undirected_edge_df = undirected_edges_df.groupby(['src', 'dest']).filter(lambda x: len(x) == 1)
        
        # iterate over the rows and add them to the edge dict if the fraction is >= min_fraction
        for index, row in single_undirected_edge_df.iterrows():
            if row['fraction'] >= min_fraction:
                # create the edge string
                edge_str = f"{row['src']} {row['edge_type']} {row['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = row['fraction']
                pass

        # now lets process the undirected edges that have multiple rows
        multiple_undirected_edges_df = undirected_edges_df.groupby(['src', 'dest']).filter(lambda x: len(x) > 1)
        # iterate over two rows, and if the sum of fraction is greater than min_fraction then keep the edge
        for i in range(0, len(multiple_undirected_edges_df), 2):
            row_pairs = multiple_undirected_edges_df.iloc[i:i+2]
            # get the sum of the fraction
            fraction_sum = row_pairs['fraction'].sum()
            # check if the sum is greater than min_fraction
            if fraction_sum >= min_fraction:
                # create the edge string
                # we use the first row to create the edge string since it has the highest fraction
                edge_str = f"{row_pairs.iloc[0]['src']} {row_pairs.iloc[0]['edge_type']} {row_pairs.iloc[0]['dest']}"
                # add to the dictionary
                selected_edges[edge_str] = float(fraction_sum)
                pass
            pass

        return list(selected_edges.keys())

    def read_prior_file(self, prior_file) -> list:
        """
        Read a prior file and return the contents as a list of strings
        Args:
            prior_file - string with the path to the prior file
            
        Returns:
            list - list of strings representing the contents of the prior file
        """
        if not os.path.exists(prior_file):
            raise FileNotFoundError(f"Prior file {prior_file} not found.")
        
        with open(prior_file, 'r') as f:
            self.prior_lines = f.readlines()
        
        return self.prior_lines

    def extract_knowledge(self, prior_lines) -> dict:
        """
        returns the knowledge from the prior file
        Args:
            prior_lines - list of strings representing the lines in the prior file
        Returns:
            dict - a dictionary where keys are
                addtemporal, forbiddirect, requiredirect
                 
                For addtemporal is a dictionary where the keys are the tier numbers (0 based) and 
                values are lists of the nodes in that tier.

                For forbiddirect and requiredirect, they will be empty in this case as this method is only for addtemporal.
        """
        tiers = {}
        inAddTemporal = False
        stop = False
        for line in prior_lines:
            # find the addtemporal line
            if line.startswith('addtemporal'):
                inAddTemporal = True
                continue
            # find the end of the addtemporal block
            if inAddTemporal and (line.startswith('\n') or line.startswith('forbiddirect')):
                inAddTemporal = False
                continue
            if inAddTemporal:
                # expect 1 binge_lag vomit_lag panasneg_lag panaspos_lag pomsah_lag

                # split the line
                line = line.strip()
                items = line.split()

                # add to dictionary
                if len(items) != 0:
                    tiers[int(items[0])-1] = items[1:]

        knowledge = {
            'addtemporal': tiers
        }

        return knowledge   

    def load_knowledge(self, knowledge:dict):
        """
        Load the knowledge
        
        The standard prior.txt file looks like this:
        
        /knowledge

        addtemporal
        1 Q2_exer_intensity_ Q3_exer_min_ Q2_sleep_hours_ PANAS_PA_ PANAS_NA_ stressed_ Span3meanSec_ Span3meanAccuracy_ Span4meanSec_ Span4meanAccuracy_ Span5meanSec_ Span5meanAccuracy_ TrailsATotalSec_ TrailsAErrors_ TrailsBTotalSec_ TrailsBErrors_ COV_neuro_ COV_pain_ COV_cardio_ COV_psych_
        2 Q2_exer_intensity Q3_exer_min Q2_sleep_hours PANAS_PA PANAS_NA stressed Span3meanSec Span3meanAccuracy Span4meanSec Span4meanAccuracy Span5meanSec Span5meanAccuracy TrailsATotalSec TrailsAErrors TrailsBTotalSec TrailsBErrors COV_neuro COV_pain COV_cardio COV_psych

        forbiddirect

        requiredirect
        
        The input dict will have the keys of addtemporal, forbiddirect, requiredirect
        
        For the addtemporal key, the value will be another dict with the keys of 1, 2, 3, etc.
        representing the tiers. The values will be a list of the nodes in that tier.
        
        Args:
        search - search object
        knowledge - dictionary with the knowledge
        
        """
        
        # check if addtemporal is in the knowledge dict
        if 'addtemporal' in knowledge:
            tiers = knowledge['addtemporal']
            for tier, nodes in tiers.items():
                # tier is a number, tetrad uses 0 based indexing so subtract 1
                for node in nodes:
                    self.add_to_tier(tier, node)
                    pass

        # if there are other knowledge types, load them here
        pass

    def extract_edges(self, text):
        """
        Extract out the edges between Graph Edges and Graph Attributes
        """
        edges = set()
        nodes = set()
        pairs = set()  # alphabetical order of nodes of an edge
        # get the lines
        lines = text.split('\n')
        startFlag=False  # True when we are in the edges, False when not
        for line in lines:
            # check if line begins with a number and period
            # convert line to python string
            line = str(line)
            if re.match(r"^\d+\.", line):
            # if startFlag == False:
            #     if "Graph Edges:" in line:
            #         startFlag = True
            #         continue  # continue to next line
            # if startFlag == True:
                # # check if there is edge information a '--'
                # if '-' in line:
                    # this is an edge so add to the set
                    # strip out the number in front  1. drinks --> happy
                    # convert to a string
                    linestr = str(line)
                    clean_edge = linestr.split('. ')[1]
                    edges.add(clean_edge)
                    
                    # add nodes
                    nodeA = clean_edge.split(' ')[0]
                    nodes.add(nodeA)
                    nodeB = clean_edge.split(' ')[2]
                    nodes.add(nodeB)
                    combined_string = ''.join(sorted([nodeA, nodeB]))
                    pairs.add(combined_string)
                    pass
        
        return list(edges), list(nodes), list(pairs)  

    def summarize_estimates(self, df):
        """
        Summarize the estimates
        """
        # get the Estimate column from the df 
        estimates = df['Estimate']       
        # get the absolute value of the estimates
        abs_estimates = estimates.abs()
        # get the mean of the absolute values
        mean_abs_estimates = abs_estimates.mean()
        # get the standard deviation of the absolute values
        std_abs_estimates = abs_estimates.std()
        return {'mean_abs_estimates': mean_abs_estimates, 'std_abs_estimates': std_abs_estimates}
        
    def edges_to_lavaan(self, edges, exclude_edges = ['---','<->','o-o']):
        """
        Convert edges to a lavaan string
        """
        lavaan_model = ""
        for edge in edges:
            nodeA = edge.split(' ')[0]
            nodeB = edge.split(' ')[2]
            edge_type = edge.split(' ')[1]
            if edge_type in exclude_edges:
                continue
            # remember that for lavaan, target ~ source
            lavaan_model += f"{nodeB} ~ {nodeA}\n"
        return lavaan_model
    
    def run_semopy(self, lavaan_model, data):  
        
        """
        run sem using semopy package
        
        lavaan_model - string with lavaan model
        data - pandas df with data
        """
        
        # create a sem model   
        model = semopy.Model(lavaan_model)

        ## TODO - check if there is a usable model,
        ## for proj_dyscross2/config_v2.yaml - no direct edges!
        ## TODO - also delete output files before writing to them so that
        ## we don't have hold overs from prior runs.
        opt_res = model.fit(data)
        estimates = model.inspect()
        stats = semopy.calc_stats(model)
        
        # change column names lval to dest and rval to src
        estimatesRenamed = estimates.rename(columns={'lval': 'dest', 'rval': 'src'})
        # convert the estimates to a dict using records
        estimatesDict = estimatesRenamed.to_dict(orient='records')        

        return ({'opt_res': opt_res,
                 'estimates': estimates, 
                 'estimatesDict': estimatesDict,
                 'stats': stats,
                 'model': model})
        
    # def run_model_search(self, df, model='gfci', 
    #                      knowledge=None, 
    #                      score=None,
    #                      test=None):
    def run_model_search(self, df, **kwargs):
        """
        Run a search
        
        Args:
        df - pandas dataframe
        
        kwargs:
        model - string with the model to use, default gfci
        knowledge - dictionary with the knowledge
        score - dictionary with the arguments for the score
            e.g. {"sem_bic": {"penalty_discount": 1}}
            
        test - dictionary with the arguments for the test alpha 
        
        Returns:
        result - dictionary with the results
        """
    
        model = kwargs.get('model', 'gfci')
        knowledge = kwargs.get('knowledge', None)
        score = kwargs.get('score', None)
        test = kwargs.get('test', None)
        depth = kwargs.get('depth', -1)
        verbose = kwargs.get('verbose', False)
        max_degree = kwargs.get('max_degree', -1)
        max_disc_path_length = kwargs.get('max_disc_path_length', -1)
        complete_rule_set_used = kwargs.get('complete_rule_set_used', False)
        guarantee_pag = kwargs.get('guarantee_pag', False)
        
        
        # load the data into the TetradSearch object
        self.load_df_into_ts(df)
        
        # check if score is not None
        if score is not None:  
            ## Use a SEM BIC score
            if 'sem_bic' in score:
                penalty_discount = score['sem_bic']['penalty_discount']
                res =self.use_sem_bic(penalty_discount=penalty_discount)
                
        if test is not None:
            if 'fisher_z' in test:
                alpha = test['fisher_z'].get('alpha',.01)
                self.use_fisher_z(alpha=alpha)
            
        # check if depth is not None - set in run_gfci
        # if depth != -1:
        #     self.set_depth(depth)
            
        if knowledge is not None:
            self.load_knowledge(knowledge)
        
        # set the verbosity
        if verbose == False:
            self.params.set(Params.VERBOSE, False)

        ## Run the selected search
        if model == 'fges':
            x = self.run_fges()
            
        elif model == 'gfci':   
                
            x = self.run_gfci(
                                depth=depth,
                                max_degree=max_degree,
                                max_disc_path_length=max_disc_path_length,
                                complete_rule_set_used=complete_rule_set_used,        
                                guarantee_pag=guarantee_pag,
                                )


        soutput = self.get_string()
        setEdges, setNodes, setPairs = self.extract_edges(soutput)
        
        result = {  'edges': list(setEdges),
                    'setEdges': setEdges, 
                    'setNodes': setNodes, 
                    'setPairs': setPairs,
                    'raw_output': soutput
                  } 
        
        return result

    def add_sem_results_to_graph(self, obj, df, format: bool = True):
        """
        Add the semopy results to the graph object.
        Args:
            obj (DgraphFlex): The graph object to add the results to.
            df (pd.DataFrame): The semopy results dataframe.
            format (bool): Whether to format the estimates or not. Defaults to True.
        
        """
        # iterate over the estimates id the df semopy output
        # and add them to the graph object

        # iterate over the estimates and add them to the graph
        # object
        
        for index, row in df.iterrows():
            if row['op'] == '~':
                source = row['rval']
                target = row['lval']
                estimate = row['Estimate']
                pvalue = row['p-value']
                # modify the edge in the existing graph obj
                # set the color based on the sign of estimate, green for positive, red for negative
                color = 'green' if estimate > 0 else 'red'
                
                obj.modify_existing_edge(source, target, color=color, strength=estimate, pvalue=pvalue)
                pass

    def get_hyper_parameters(self, df:pd.DataFrame, 
                        model: str = 'gfci',
                        target_fpr: float = 0.01,
                        n_permutations: int = None,
                        pd_values: List = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],
                        alpha_values: List = [.05,.01,.005,.001,.0005,.0001],
                        knowledge: dict = {},
                        lag_flag: int = 0, # lag the data if 1
                        lag_stub: str = '_',
                        verbose: int = 1
                        ) -> dict:
        """
        
        Determine the hyperparameters for a given model
        using permutation analysis.
        This is a brute force approach to hyperparameter
        model search. Given either single or multiple pd_values 
        or alpha_values, the variable with the single value will
        be used as the fixed value and the other variable will
        be varied over the range of values.
        The function will then compute the false positive rate
        (FPR) for each combination of pd_value and alpha_value, 
        returning the best combination that is <= target_fpr.
        
        Arguments:
        df : pd.DataFrame : the dataframe to use
        model : str : the model to use, default 'gfci'
        target_fpr : float : target false positive rate
        n_permutations : int : number of permutations to run
        pd_values : List : penalty discount values to use - default [1.0]
        alpha_values : List : list of alpha values to use - default [.05,.01,.005,.001,.0005,.0001]
        knowledge : dict : knowledge to use
        lag_flag : int : if 1, add lagged columns to the dataframe
        lag_stub : str : string to append to the column names for the lagged variables
        verbose : int : verbosity level, default 1

        Returns:
        dict : dictionary of results
        
        """
        
        # check if in jupyter to select the progress bar code
        if self.in_jupyter():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm  
            


        # calculate the number of permutations based on the target_fpr
        if n_permutations is None:
            # calculate the number of permutations based on the target_fpr
            n_permutations = int(1 / (target_fpr/10.0))
            if verbose: print(f"Number of permutations set to {n_permutations} based on target fpr of {target_fpr}.")

        # dict to hold the results of permutation analysis
        permuted_results = {}  # keys are the alpha values
        
        # create n_permutations
        permuted_dfs = self.create_permuted_dfs(df, n_permutations=n_permutations)
        
        # iterate over the penalty_discount values and alpha values
        for alpha_value in alpha_values:
            for pd_value in pd_values:
                penalty_discount = pd_value
                combined_key = f"pd-{penalty_discount}_alpha-{alpha_value}"
                # check if the penalty discount is already in the results
                if verbose >1: print(f"Permutation with penalty discount {penalty_discount} and alpha {alpha_value}...")
                # create key for this penalty discount
                permuted_results[combined_key] = {"data": []}
                pass
            
                sum_fpr = 0.0
                
                # iterate over the permuted dataframes
                #for permuted_df in permuted_dfs:
                for i in tqdm(range(n_permutations), desc=f"Finding fpr for pd {pd_value} alpha {alpha_value}", unit="perm"):
                    
                    permuted_df = permuted_dfs[i]
                    
                    if lag_flag == 1:
                        # add lag columns
                        permuted_df = self.add_lag_columns(permuted_df, lag_stub=lag_stub)
                    # standardize the data
                    permuted_df = self.standardize_df_cols(permuted_df)
                    # run the search
                    searchResult = self.run_model_search(permuted_df, model='gfci', 
                                                        knowledge=knowledge, 
                                                        score={'sem_bic': {'penalty_discount': penalty_discount}},
                                                        test={'fisher_z': {'alpha': alpha_value}},
                                                        verbose=False)
                    # create dictionary for result to include penalty_discount
                    
                    # calculate fpr
                    total_possible_edges = (len(permuted_df.columns) * (len(permuted_df.columns) - 1)) / 2
                    fpr = len(searchResult['setEdges']) / total_possible_edges
                    sum_fpr += fpr
                    
                    info = {
                        'penaltyDiscount': penalty_discount,
                        'alphaValue': alpha_value,
                        'numEdges': len(searchResult['setEdges']),
                        'edges': list(searchResult['setEdges']),
                        'numCols': len(permuted_df.columns),
                        'fpr': fpr
                    }
                    permuted_results[combined_key]['data'].append(info)
                    if verbose >2: print(f"\n pd:{penalty_discount} alpha:{alpha_value} with fpr {fpr} complete!")
                
                # calculate the avg fpr
                avg_fpr = sum_fpr / n_permutations
                # store in the results dict
                permuted_results[combined_key]['avg_fpr'] = avg_fpr
                permuted_results[combined_key]['alpha'] = alpha_value
                permuted_results[combined_key]['pd'] = pd_value

                if verbose>1: print(f"Avg fpr for penalty discount {penalty_discount} and alpha {alpha_value} is {avg_fpr}")
                
                # initialize the result dict
                result = {
                    'model': model,
                    'calc_fpr': target_fpr,
                    'n_permutations': n_permutations,
                }

                # check if avg_fpr is less than or equal to target_fpr
                # means that we covered the range needed to calcuate the best value
                if avg_fpr <= target_fpr:
                    #
                    # use linear interpolation to find the best value

                    # get the current and previous values for fpr and pd
                    # from the permuted_results
                    # Get a list of items (key-value pairs)
                    items_list = list(permuted_results.items())

                    # Get the last entry (key-value pair)
                    last_entry = items_list[-1]
                    # Get the next-to-last entry (key-value pair)
                    next_to_last_entry = items_list[-2]

                    # get the last and next to last values for fpr, pd and alpha
                    last_fpr = last_entry[1]['avg_fpr']
                    last_pd = last_entry[1]['pd']
                    next_to_last_fpr = next_to_last_entry[1]['avg_fpr']
                    next_to_last_pd = next_to_last_entry[1]['pd']
                    last_alpha = last_entry[1]['alpha']
                    next_to_last_alpha = next_to_last_entry[1]['alpha']

                    if len(alpha_values) == 1:
                        # calculate the slope
                        slope = (np.log10(last_fpr) - np.log10(next_to_last_fpr)) / (last_pd - next_to_last_pd)
                        # calculate the intercept
                        intercept = np.log10(last_fpr) - (slope * last_pd)
                        # calculate the best pd value
                        best_pd = (float(np.log10(target_fpr)) - intercept) / slope
                        result.update(
                            {
                            'penaltyDiscount': float(best_pd),
                            'alphaValue': last_alpha,
                            'run_results': permuted_results
                            }
                        )
                        return result

                    elif len(pd_values) == 1:
                        # calculate the slope for calculating the best alpha value
                        slope = (np.log10(last_fpr) - np.log10(next_to_last_fpr)) / (last_alpha - next_to_last_alpha)
                        # calculate the intercept
                        intercept = np.log10(last_fpr) - (slope * last_alpha)
                        # calculate the best alpha value
                        best_alpha = (np.log10(target_fpr) - intercept) / slope
                        result.update(
                            {
                            'penaltyDiscount': last_pd,
                            'alphaValue': best_alpha,
                            'run_results': permuted_results
                            }
                        )
                        return result
                    else:
                        result = None
                        raise ValueError("Both pd_values and alpha_values cannot be lists.")
                    break
                pass
            pass


    def in_jupyter(self)->bool:
        """
        Check if the code is running in a Jupyter Notebook environment.

        Returns:
            bool: True if running in a Jupyter Notebook, False otherwise.
        
        """
        try:
            # Check if the IPython module is available
            from IPython import get_ipython
            ipython = get_ipython()
            
            # Check if the current environment is a Jupyter Notebook
            if ipython is not None and 'IPKernelApp' in ipython.config:
                return True
        except ImportError:
            pass
        
        return False

if __name__ == "__main__":
    # Example usage of MyTetradSearch
    


    # Create an instance of MyTetradSearch
    ts = MyTetradSearch()

    # load a dataframe for testing
    df_file = "pytetrad_plus/boston_data_raw.csv"
    df_raw = pd.read_csv(df_file)
    
    if df_raw.empty:
        raise ValueError(f"Failed to load the DataFrame from {df_file}. Please check the file.")
    else:
        # add the lagged columns to the DataFrame
        df = ts.add_lag_columns(df_raw, lag_stub='_lag')
        # standardize the data
        df = ts.standardize_df_cols(df)
        # Load the DataFrame into the TetradSearch object
        ts.load_df_into_ts(df)

    # read the prior file
    prior_lines = ts.read_prior_file('pytetrad_plus/boston_prior.txt')
    # extract knowledge from the prior lines
    knowledge = ts.extract_knowledge(prior_lines)
    # load the knowledge into the TetradSearch object
    ts.load_knowledge(knowledge)


    ## Run the search
    searchResult = ts.run_model_search(df, model='gfci', 
                                            knowledge=knowledge, 
                                            score={'sem_bic': {'penalty_discount': 1.0}},
                                            test={'fisher_z': {'alpha': .01}})
    
    
    lavaan_model = ts.edges_to_lavaan(searchResult['setEdges'])
    
    # run semopy
    sem_results = ts.run_semopy(lavaan_model, df)
    
    # plot into png
    png_path = 'pytetrad_plus/boston_data.png'
    g = semopy.semplot(sem_results['model'], png_path,  plot_covs = True)

    # get the estmates
    estimates_sem = sem_results['estimates']
    
    # summary of the estimates
    estimates = ts.summarize_estimates(estimates_sem)
    
    result = {'setEdges': list(searchResult['setEdges']), 
                'setNodes': list(searchResult['setNodes']), 
                'setPairs': list(searchResult['setPairs']), 
                'ESMean': estimates['mean_abs_estimates'],
                'ESStd': estimates['std_abs_estimates'],
                'estimatesSEM': sem_results['estimatesDict']
                } 
 
    # write the result to a json file
    with open('pytetrad_plus/boston_result.json','w') as f:
        json.dump(result, f, indent=4)

    # run the hyperparameter search
    print("Running hyperparameter search...")
    # create the df for the hyperparameter search
    df_raw_std = ts.standardize_df_cols(df_raw)
    results = ts.get_hyper_parameters(
                    df_raw_std,
                    alpha_values = [0.01],
                    knowledge=knowledge,
                    target_fpr= 0.005,
                    lag_flag=True,
                    lag_stub='_lag',
                    verbose=2,

                )
    
    # save the results to a file

    file_out = f"pytetrad_plus/boston_hyperparameter_results.json"
    with open(file_out, 'w') as f:
        json.dump(results, f, indent=4)
        pass  # assign the method for testing

    # run the model search with the best hyperparameters

    searchResult = ts.run_model_search(df, model='gfci', 
                                            knowledge=knowledge, 
                                            #score={'sem_bic': {'penalty_discount': results['penaltyDiscount']}},
                                            #test={'fisher_z': {'alpha': results['alphaValue']}}
                                            score={'sem_bic': {'penalty_discount': 1.1}},
                                            test={'fisher_z': {'alpha': .01}}
                                            )
    
    
    lavaan_model = ts.edges_to_lavaan(searchResult['setEdges'])
    
    # run semopy
    sem_results = ts.run_semopy(lavaan_model, df)
    
    # plot into png
    png_path = 'pytetrad_plus/boston_data_optimum_hp.png'
    g = semopy.semplot(sem_results['model'], png_path,  plot_covs = True)

    # get the estmates
    estimates_sem = sem_results['estimates']
    
    pass
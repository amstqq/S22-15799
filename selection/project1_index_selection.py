import copy
import json
import logging
import pickle
import sys
import time
import os

import pandas as pd
from sql_metadata import Parser

from .algorithms.anytime_algorithm import AnytimeAlgorithm
from .algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from .algorithms.db2advis_algorithm import DB2AdvisAlgorithm
from .algorithms.dexter_algorithm import DexterAlgorithm
from .algorithms.drop_heuristic_algorithm import DropHeuristicAlgorithm
from .algorithms.extend_algorithm import ExtendAlgorithm
from .algorithms.relaxation_algorithm import RelaxationAlgorithm
from .benchmark import Benchmark
from .dbms.postgres_dbms import PostgresDatabaseConnector
from .selection_algorithm import AllIndexesAlgorithm, NoIndexAlgorithm
from .workload_generator import WorkloadGenerator


ALGORITHMS = {
    "anytime": AnytimeAlgorithm,
    "auto_admin": AutoAdminAlgorithm,
    "db2advis": DB2AdvisAlgorithm,
    "dexter": DexterAlgorithm,
    "drop": DropHeuristicAlgorithm,
    "extend": ExtendAlgorithm,
    "relaxation": RelaxationAlgorithm,
    "no_index": NoIndexAlgorithm,
    "all_indexes": AllIndexesAlgorithm,
}

DEFAULT_DB = "project1db"
DEFAULT_USER = "project1user"
DEFAULT_PASS = "project1pass"
CONFIG_PATH = "./project1.json"


class P1IndexSelection:
    def __init__(self, workload_csv_path, log_level=None, disable_output_files=True):
        if "CRITICAL_LOG" == log_level:
            logging.getLogger().setLevel(logging.CRITICAL)
        if "ERROR_LOG" == log_level:
            logging.getLogger().setLevel(logging.ERROR)
        if "INFO_LOG" == log_level:
            logging.getLogger().setLevel(logging.INFO)

        # Disable output files generated by benchmark class
        self.disable_output_files = disable_output_files

        logging.debug("Init IndexSelection")
        self.db_connector = None
        self.workload = None
        self.config_file = CONFIG_PATH
        self.db_name = DEFAULT_DB
        self.db_user = DEFAULT_USER
        self.db_pass = DEFAULT_PASS
        self.workload_csv_path = workload_csv_path

        # TODO: Add other paths
        if "epinions" in workload_csv_path:
            self.workload_name = "epinions"
        self.workload_name = "epinions"

        # Set up Workload generator which reads workload_csv
        self.workload_generator = WorkloadGenerator(
            self.workload_csv_path, sample_size=1000)

        print(f"Running on benchmark {self.workload_name}...")

    def run(self, debug=False):
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Starting Index Selection Evaluation")

        self._run_algorithms()

    def _run_algorithms(self):
        with open(self.config_file) as f:
            config = json.load(f)
        # Initialize postgres connector
        self.setup_db_connector()
        # Enable hypopg
        self.db_connector.enable_simulation()
        # Show current indexes
        self.db_connector.show_curr_indexes()
        # TODO: REMOVE DROP INDEX, or print existing indexes
        self.db_connector.drop_indexes()
        # TODO: remove!
        self.db_connector.show_curr_indexes()

        # Set the random seed to obtain deterministic statistics (and cost estimations)
        # because ANALYZE (and alike) use sampling for large tables
        self.db_connector.create_statistics()
        self.db_connector.commit()

        print("Setup complete!\nStart running algorithms...")

        total_runtime = time.time()

        # Obtain best index from each algorithm
        best_indexes_all_algorithms = []
        number_of_actual_runs = config[
            "number_of_actual_runs"] if "number_of_actual_runs" in config else 0
        for algorithm_config in config["algorithms"]:
            algo_start_time = time.time()
            # There are multiple configs if there is a parameter list
            # configured (as a list in the .json file)
            algorithm_config["number_of_actual_runs"] = number_of_actual_runs
            configs = self._find_parameter_list(algorithm_config)

            best_cost = float('inf')
            best_runtime = float('inf')
            best_indexes = None
            for algorithm_config_unfolded in configs:
                start_time = time.time()
                # Generate new sample workload of sample_size=100
                self.workload = self.workload_generator.generate_sample_workload()
                cfg = algorithm_config_unfolded
                indexes, what_if, cost_requests, cache_hits = self._run_algorithm(
                    cfg)

                calculation_time = round(time.time() - start_time, 2)
                benchmark = Benchmark(
                    self.workload,
                    indexes,
                    self.db_connector,
                    algorithm_config_unfolded,
                    calculation_time,
                    self.disable_output_files,
                    config,
                    cost_requests,
                    cache_hits,
                    what_if,
                )

                cost, runtime, hit = benchmark.benchmark()

                # Update best_indexes
                # If "number_of_actual_runs" > 0, then actual runtime is returned from benchmark
                if "number_of_actual_runs" in cfg and cfg["number_of_actual_runs"] > 0:
                    if runtime < best_runtime:
                        best_indexes = indexes
                        best_runtime = runtime
                        best_cost = cost
                        best_indexes = indexes
                else:
                    if cost < best_cost:
                        best_indexes = indexes
                        best_cost = cost
                        best_runtime = 0
                        best_indexes = indexes
            best_indexes_all_algorithms.append(
                (algorithm_config["name"], best_runtime, best_cost, best_indexes))
            algo_time = round(time.time() - algo_start_time, 2)
            print(
                f"{algorithm_config['name']} finished in {algo_time} seconds...")

        # Store all indexes in sorted order, weight cost by sample runtime
        # Weighted cost = cost * (1 + runtime/max_runtime)
        # Runtime = 0 if self.number_of_actual_runs = 0, ie. no sample query is run
        # Cost is computed by cost estimator. Runtime is obtained from running sample queries.
        max_runtime = max([x[1] for x in best_indexes_all_algorithms]
                          )
        if max_runtime == 0:
            max_runtime = 1
        max_cost = max([x[2] for x in best_indexes_all_algorithms])
        if max_cost == 0:
            max_cost = 1

        best_indexes_all_algorithms.sort(
            key=lambda x: x[2]/max_cost+x[1]/max_runtime)

        total_runtime = round(time.time() - total_runtime, 2)
        print(f"Index Selection Finished in {total_runtime} seconds")

        self.save_indexes(best_indexes_all_algorithms)
        self.print_indexes(best_indexes_all_algorithms)
        self.write_actions_sql_file(
            best_indexes_all_algorithms[0][3], './actions.sql')

    @staticmethod
    def print_indexes(best_indexes):
        max_runtime = max([x[1] for x in best_indexes]
                          )
        if max_runtime == 0:
            max_runtime = 1
        max_cost = max([x[2] for x in best_indexes])
        if max_cost == 0:
            max_cost = 1
        for name, runtime, cost, indexes in best_indexes:
            print(
                f"====== {name}, cost: {cost:.4f}, runtime: {runtime:.4f}, weighted_cost: {cost/max_cost+runtime/max_runtime:.4f} ======= \nIndexes: {indexes}\n")

    def save_indexes(self, indexes):
        save_dir = os.path.join(os.getcwd(), f"indexes_results/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        iteration = 1
        filepath = os.path.join(
            save_dir, f"{self.workload_name}_{iteration}.pickle")
        while os.path.exists(filepath):
            iteration += 1
            filepath = os.path.join(
                save_dir, f"{self.workload_name}_{iteration}.pickle")

        with open(filepath, "ba") as file:
            pickle.dump(indexes, file)

    def load_indexes(self):
        save_dir = os.path.join(os.getcwd(), f"indexes_results/")
        if not os.path.exists(save_dir):
            return None

        filenames = os.listdir(save_dir)

        most_recent_filenames = sorted([
            filename for filename in filenames if self.workload_name in filename], reverse=True)

        if len(most_recent_filenames) == 0:
            return None

        most_recent_filename = most_recent_filenames[0]
        with open(os.path.join(save_dir, most_recent_filename), "rb") as file:
            indexes = pickle.load(file)

        return indexes

    def write_actions_sql_file(self, indexes: list, filename: str) -> list:
        print("Generating actions.sql...")
        with open('./actions.sql', 'w') as file:
            for i, index in enumerate(indexes):
                table_name = index.table()
                statement = (
                    f"create index {index.index_idx()} "
                    f"on {table_name} ({index.joined_column_names()});"
                )
                file.write(statement)
                file.write('\n')

    # Parameter list example: {"max_indexes": [5, 10, 20]}
    # Creates config for each value
    def _find_parameter_list(self, algorithm_config):
        parameters = algorithm_config["parameters"]
        configs = []
        if parameters:
            # if more than one list --> raise
            self.__check_parameters(parameters)
            for key, value in parameters.items():
                if isinstance(value, list):
                    for i in value:
                        new_config = copy.deepcopy(algorithm_config)
                        new_config["parameters"][key] = i
                        configs.append(new_config)
        if len(configs) == 0:
            configs.append(algorithm_config)
        return configs

    def __check_parameters(self, parameters):
        counter = 0
        for key, value in parameters.items():
            if isinstance(value, list):
                counter += 1
        if counter > 1:
            raise Exception("Too many parameter lists in config")

    def _run_algorithm(self, config):
        # TODO: REMOVE DROP INDEX
        # TODO: Print already existent indexes before dropping
        self.db_connector.drop_indexes()
        self.db_connector.commit()
        self.setup_db_connector()

        algorithm = self.create_algorithm_object(
            config["name"], config["parameters"])
        logging.info(f"Running algorithm {config}")
        indexes = algorithm.calculate_best_indexes(self.workload)
        logging.info(f"Indexes found: {indexes}")
        what_if = algorithm.cost_evaluation.what_if

        cost_requests = (
            self.db_connector.cost_estimations
            if config["name"] == "db2advis"
            else algorithm.cost_evaluation.cost_requests
        )
        cache_hits = (
            0 if config["name"] == "db2advis" else algorithm.cost_evaluation.cache_hits
        )
        return indexes, what_if, cost_requests, cache_hits

    def create_algorithm_object(self, algorithm_name, parameters) -> AllIndexesAlgorithm:
        algorithm = ALGORITHMS[algorithm_name](self.db_connector, parameters)
        return algorithm

    def setup_db_connector(self):
        if self.db_connector:
            logging.info("Create new database connector (closing old)")
            self.db_connector.close()
        self.db_connector = PostgresDatabaseConnector(
            self.db_name, self.db_user, self.db_pass)

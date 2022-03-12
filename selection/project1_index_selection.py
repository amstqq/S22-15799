import copy
import json
import logging
import pickle
import sys
import time
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
from .workload import Workload, Column, Table, Query

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

EPINIONS_SCHEMA = {
    "item": set(["i_id", "title"]),
    "useracct": set(["u_id", "name"]),
    "review": set(["a_id", "u_id", "i_id", "rating", "rank"]),
    "trust": set(["source_u_id", "target_u_id", "trust", "creation_date"]),
    "review_rating": set(["u_id", "a_id", "rating", "status", "creation_date", "last_mod_date", "type", "vertical_id"])
}

CONFIG_PATH = "./project1.json"


class P1IndexSelection:
    def __init__(self, workload_csv_path, log_level=None, disable_output_files=True):
        if "CRITICAL_LOG" == log_level:
            logging.getLogger().setLevel(logging.CRITICAL)
        if "ERROR_LOG" == log_level:
            logging.getLogger().setLevel(logging.ERROR)
        if "INFO_LOG" == log_level:
            logging.getLogger().setLevel(logging.INFO)

        self.disable_output_files = disable_output_files

        logging.debug("Init IndexSelection")
        self.db_connector = None
        self.config_file = CONFIG_PATH
        self.db_name = DEFAULT_DB
        self.db_user = DEFAULT_USER
        self.db_pass = DEFAULT_PASS
        self.workload_csv_path = workload_csv_path

        self.tables = {}
        self.columns = {}  # key=table_name, value={column_name: column_object}
        self.workload = None

    def run(self, debug=False):
        """This is called when running `python3 -m selection`.
        """
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Starting Index Selection Evaluation")

        self._run_algorithms()

    def _build_workload_from_csv(self, csv_path):
        queries = []
        df = pd.read_csv(csv_path, header=None, usecols=[7, 13], names=[
                         "query_type", "query_text"])
        # TODO: get rid of this later
        df = df[:500]

        for index, row in df.iterrows():
            if pd.isna(row["query_type"]):
                continue
            query = self._build_query_object(
                index, row["query_text"], EPINIONS_SCHEMA)
            if query is not None:
                queries.append(query)
        self.workload = Workload(queries)

        # Print info about the query
        # for query in self.workload.queries:
        #     columns = [
        #         f"{column.table.name}.{column.name}" for column in query.columns]
        #     logging.debug(f"Q{query.nr}, columns={columns}")

    def _build_query_object(self, index: int, query_str: str, db_schema: dict):
        if "pg_" in query_str:
            return None

        # Remove "execute <unamed>:"
        query_str = query_str[query_str.find(':')+1:]

        # Ignore non-relevant queries
        if "BEGIN" in query_str or "COMMIT" in query_str:
            return None

        try:
            p = Parser(query_str)
            tables = p.tables
            columns = p.columns
        except:
            return None

        # Skip queries that dont have a where clause
        if len(tables) == 0 or len(columns) == 0:
            return None

        # Extract list of columns referenced in this query
        added = False
        column_objects = []
        for column in columns:
            if column == "*":
                continue

            if "." in column:
                table_name, column_name = column.split(".", 1)
                if table_name in self.columns and column_name in self.columns[table_name]:
                    column_objects.append(
                        self.columns[table_name][column_name])
                    added = True
            else:
                # Find which table this column corresponds to
                for table_name in tables:
                    if column in db_schema[table_name]:
                        added = True
                        column_objects.append(
                            self.columns[table_name][column])
                        break

                # Cannot find a corresponding table for this column
                if not added:
                    logging.debug("Invalid Query: " + query_str)
                    return None

        query = Query(index, query_str+";", column_objects)
        return query

    def _run_algorithms(self):
        with open(self.config_file) as f:
            config = json.load(f)
        self.setup_db_connector()
        # TODO: DROP INDEX
        self.db_connector.drop_indexes()

        # Construct table object with columns
        for table_name, column_names in EPINIONS_SCHEMA.items():
            table_obj = Table(table_name)
            self.tables[table_name] = table_obj
            self.columns[table_name] = {}
            for column_name in column_names:
                column_obj = Column(column_name)
                column_obj.table = table_name
                table_obj.add_column(column_obj)
                self.columns[table_name][column_name] = column_obj

        self._build_workload_from_csv(self.workload_csv_path)

        # Set the random seed to obtain deterministic statistics (and cost estimations)
        # because ANALYZE (and alike) use sampling for large tables
        self.db_connector.create_statistics()
        self.db_connector.commit()

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

        # Print all indexes in sorted order
        best_indexes_all_algorithms.sort(key=lambda x: x[1])
        for name, runtime, cost, indexes in best_indexes_all_algorithms:
            print(
                f"====== {name}, cost: {cost:.4f}, runtime: {runtime:.4f} ======= \nIndexes: {indexes}\n")

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
        # TODO: DROP INDEX
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

    def _parse_command_line_args(self):
        arguments = sys.argv
        if "CRITICAL_LOG" in arguments:
            logging.getLogger().setLevel(logging.CRITICAL)
        if "ERROR_LOG" in arguments:
            logging.getLogger().setLevel(logging.ERROR)
        if "INFO_LOG" in arguments:
            logging.getLogger().setLevel(logging.INFO)
        if "DISABLE_OUTPUT_FILES" in arguments:
            self.disable_output_files = True
        for argument in arguments:
            if ".json" in argument:
                return argument

    def setup_db_connector(self):
        if self.db_connector:
            logging.info("Create new database connector (closing old)")
            self.db_connector.close()
        self.db_connector = PostgresDatabaseConnector(
            self.db_name, self.db_user, self.db_pass)

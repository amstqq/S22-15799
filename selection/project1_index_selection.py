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


class P1IndexSelection:
    def __init__(self, workload_csv_path):
        logging.debug("Init IndexSelection")
        self.db_connector = None
        self.config_file = "./project1.json"
        self.db_name = DEFAULT_DB
        self.db_user = DEFAULT_USER
        self.db_pass = DEFAULT_PASS
        self.workload_csv_path = workload_csv_path

        self.tables = {}
        self.columns = {}  # key=table_name, value={column_name: column_object}
        self.workload = None

    def run(self, debug=True):
        """This is called when running `python3 -m selection`.
        """
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Starting Index Selection Evaluation")

        self._run_algorithms()

    def _build_workload_from_csv(self, csv_path):
        queries = []
        df = pd.read_csv(csv_path, header=None, usecols=[7, 13, 14], names=[
                         "query_type", "query_text", "params"])
        for index, row in df.iterrows():
            if pd.isna(row["query_type"]):
                continue
            query = self._build_query_object(
                index, row["query_text"], EPINIONS_SCHEMA)
            if query is not None:
                queries.append(query)
        self.workload = Workload(queries)

        # Print info about the query
        for query in self.workload.queries:
            columns = [
                f"{column.table.name}.{column.name}" for column in query.columns]
            logging.debug(f"Q{query.nr}, columns={columns}")

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
            if "." in column:
                table_name, column_name = column.split(".", 1)
                added = True
                column_objects.append(self.columns[table_name][column_name])
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

        query = Query(index, query_str, column_objects)
        return query

    def _run_algorithms(self):
        with open(self.config_file) as f:
            config = json.load(f)
        self.setup_db_connector()
        # self.db_connector.drop_indexes()

        # Construct table object
        for table_name, column_names in EPINIONS_SCHEMA.items():
            table = Table(table_name)
            self.tables[table_name] = table
            self.columns[table_name] = {}
            for column_name in column_names:
                column = Column(column_name)
                column.table = table
                table.add_column(column)
                self.columns[table_name][column_name] = column

        self._build_workload_from_csv(self.workload_csv_path)
        return

        # Set the random seed to obtain deterministic statistics (and cost estimations)
        # because ANALYZE (and alike) use sampling for large tables
        self.db_connector.create_statistics()
        self.db_connector.commit()

        for algorithm_config in config["algorithms"]:
            # CoPhy must be skipped and manually executed via AMPL because it is not
            # integrated yet.
            if algorithm_config["name"] == "cophy":
                continue

            # There are multiple configs if there is a parameter list
            # configured (as a list in the .json file)
            configs = self._find_parameter_list(algorithm_config)
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
                benchmark.benchmark()

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
        # self.db_connector.drop_indexes()
        # self.db_connector.commit()
        # self.setup_db_connector(self.database_name, self.database_system)

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

    def create_algorithm_object(self, algorithm_name, parameters):
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

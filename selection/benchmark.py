import datetime
import json
import logging
import os.path
import pickle
import random
import subprocess
import time

from .utils import s_to_ms


class Benchmark:
    def __init__(
        self,
        workload,
        indexes,
        db_connector,
        config,
        calculation_time,
        disable_output_files,
        global_config,
        cost_requests,
        cache_hits,
        what_if=None,
    ):
        self.workload = workload
        self.db_connector = db_connector
        self.indexes = indexes
        self.timeout_ms = s_to_ms(config["timeout"])
        self.number_of_runs = (
            config["number_of_actual_runs"] if "number_of_actual_runs" in config else 0
        )
        self.config = config
        self.calculation_time = calculation_time
        self.disable_output_files = disable_output_files
        self.what_if = what_if
        self.cost_requests = cost_requests
        self.cache_hits = cache_hits
        self.cost_estimation_duration = self.db_connector.cost_estimation_duration
        self.index_simulation_duration = self.db_connector.index_simulation_duration
        self.simulated_indexes = self.db_connector.simulated_indexes

        self.scale_factor = 1
        self.benchmark_name = global_config["benchmark_name"]
        self.db_system = global_config["database_system"]
        self.seed = None
        if "seed" in global_config:
            self.seed = global_config["seed"]

        self._set_filenames()

    def benchmark(self):
        # TODO: DROP INDEX
        self.db_connector.drop_indexes()

        logging.info("Benchmark with config: {}".format(self.config))
        # Number of runs can be set to 0 to get estimated workload
        # costs. Estimated sizes are returned instead of actual index sizes
        # to avoid creating the indexes.
        if self.number_of_runs > 0:
            self._create_indexes()
        else:
            self.index_create_time = 0
            for index in self.indexes:
                self.what_if.simulate_index(index, store_size=True)
        cost, runtime, hit = self._benchmark()
        if self.number_of_runs > 0:
            # TODO: DROP INDEX
            self._drop_indexes()
        else:
            self.what_if.drop_all_simulated_indexes()

        return cost, runtime, hit

    def _create_csv_header(self):
        header = [
            "date",
            "commit",
            "benchmark name",
            "algorithm name",
            "parameters",
            "sim overall costs",
            "actual overall_runtime",
            "actual overall hits",
            "algorithm runtime",
            "algorithm cost time",
            "algorithm index creation time",
            "algorithm created #indexes",
            "#indexes",
            "actual index create time",
            "actual/sim index memory consumption",
            "cost requests",
            "cache hits",
        ]
        header.append("indexed columns")
        return ";".join(header)

    def _git_hash(self):
        githash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"])
        return githash.decode("ascii").replace("\n", "")

    def _store_results(self, overall_costs, overall_runtime, overall_hits):
        config = self.config
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_hash = self._git_hash()
        indexes_size = self.db_connector.indexes_size()

        # see comment above
        if self.number_of_runs == 0:
            indexes_size = sum(
                [index.estimated_size for index in self.indexes])

        csv_entry = [
            date,
            commit_hash,
            self.benchmark_name,
            config["name"],
            config["parameters"],
            "{0:.2f}".format(overall_costs),
            "{0:.2f}".format(overall_runtime),
            "{0:.2f}".format(overall_hits),
            self.calculation_time,
            self.cost_estimation_duration,
            self.index_simulation_duration,
            self.simulated_indexes,
            len(self.indexes),
            self.index_create_time,
            indexes_size,
            self.cost_requests,
            self.cache_hits,
        ]
        csv_entry.append(sorted(self.indexes))
        self._append_to_csv(";".join([str(x) for x in csv_entry]))

        with open(self.picklename, "ba") as file:
            pickle.dump(self.indexes, file)

    def _append_to_csv(self, entry):
        header = self._create_csv_header()
        entry = entry.replace("'", '"').replace("None", "null")
        if not os.path.isfile(self.filename):
            entry = header + "\n" + entry
        with open(self.filename, "a") as f:
            f.write(entry + "\n")
        logging.info(f"Results written to {self.filename}")

    def _benchmark(self):
        logging.info("Benchmark all queries")
        overall_runtime = 0
        overall_hits = 0
        overall_costs = 0

        for query_id in range(len(self.workload.queries)):
            query = self.workload.queries[query_id]
            cost = self.db_connector.get_cost(query)
            overall_costs += cost
        for i in range(self.number_of_runs):
            logging.debug("Benchmark Run {}".format(i))
            random_query_indexes = list(range(len(self.workload.queries)))
            seed = time.time()
            if self.seed:
                seed = self.seed
            logging.debug(f"Random seed: {seed}")
            random.seed(seed)
            random.shuffle(random_query_indexes)
            for query_index in random_query_indexes:
                query = self.workload.queries[query_index]
                logging.debug("Run {}".format(query))
                execution_time, plan = self._benchmark_query(query)
                overall_runtime += execution_time
                overall_hits += self._calculate_hits(plan)
        overall_runtime = overall_runtime / \
            self.number_of_runs if self.number_of_runs > 0 else 0
        overall_hits = overall_hits / \
            (len(self.workload.queries) *
             self.number_of_runs) if (len(self.workload.queries) *
                                      self.number_of_runs) > 0 else 0

        self._store_results(overall_costs, overall_runtime, overall_hits)

        return overall_costs, overall_runtime, overall_hits

    def _benchmark_query(self, query):
        exec_result = self.db_connector.exec_query(
            query, timeout=self.timeout_ms)
        return exec_result

    def _calculate_hits(self, plan):
        ratio = None
        if "Shared Hit Blocks" in plan:
            hits = plan["Shared Hit Blocks"]
            reads = plan["Shared Read Blocks"]
            ratio = hits / (hits + reads)
        return ratio

    def _create_indexes(self):
        logging.info("Creating the indexes")
        start_time = time.time()
        for index in self.indexes:
            logging.debug("create index on {}".format(index))
            self.db_connector.create_index(index)
        self.index_create_time = round(time.time() - start_time, 2)

    def _drop_indexes(self):
        for index in self.indexes:
            self.db_connector.drop_index(index)
        self.db_connector.commit()

    def _set_filenames(self):
        if self.disable_output_files:
            self.filename = os.devnull
            self.picklename = os.devnull
            return

        identifier = (
            f"{self.config['name']}_{self.benchmark_name}"
            f"_{len(self.workload.queries)}"
        )
        self.filename = f"benchmark_results/results_{identifier}_queries.csv"
        self.picklename = f"benchmark_results/indexes_{identifier}_queries.pickle"

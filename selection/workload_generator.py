import csv
import glob
import re
import time
from typing import List
from sql_metadata import Parser
import logging

import pandas as pd
import numpy as np
import pglast
from pandarallel import pandarallel
from plumbum import cli
from .workload import Workload, Column, Table, Query


pandarallel.initialize(verbose=1)


class WorkloadGenerator:
    _PG_LOG_COLUMNS: List[str] = [
        "log_time",
        "user_name",
        "database_name",
        "process_id",
        "connection_from",
        "session_id",
        "session_line_num",
        "command_tag",
        "session_start_time",
        "virtual_transaction_id",
        "transaction_id",
        "error_severity",
        "sql_state_code",
        "message",
        "detail",
        "hint",
        "internal_query",
        "internal_query_pos",
        "context",
        "query",
        "query_pos",
        "location",
        "application_name",
        "backend_type",
    ]

    _EPINIONS_SCHEMA = {
        "item": set(["i_id", "title"]),
        "useracct": set(["u_id", "name"]),
        "review": set(["a_id", "u_id", "i_id", "rating", "rank"]),
        "trust": set(["source_u_id", "target_u_id", "trust", "creation_date"]),
        "review_rating": set(["u_id", "a_id", "rating", "status", "creation_date", "last_mod_date", "type", "vertical_id"])
    }

    """
    Convert PostgreSQL query logs into pandas DataFrame objects.
    """

    def get_dataframe(self):
        """
        Get a raw dataframe of query log data.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing the query log data.
            Note that irrelevant query log entries are still included.
        """
        return self._df

    def get_grouped_dataframe_interval(self, interval=None):
        """
        Get the pre-grouped version of query log data.

        Parameters
        ----------
        interval : pd.TimeDelta or None
            time interval to group and count the query templates
            if None, pd is only aggregated by template

        Returns
        -------
        grouped_df : pd.DataFrame
            Dataframe containing the pre-grouped query log data.
            Grouped on query template and optionally log time.
        """
        gb = None
        if interval is None:
            gb = self._df.groupby("query_template").size()
            gb.drop("", axis=0, inplace=True)
        else:
            gb = self._df.groupby("query_template").resample(interval).size()
            gb.drop("", axis=0, level=0, inplace=True)
        grouped_df = pd.DataFrame(gb, columns=["count"])
        return grouped_df

    def get_grouped_dataframe_params(self):
        """
        Get the pre-grouped version of query log data.

        Returns
        -------
        grouped_df : pd.DataFrame
            Dataframe containing the pre-grouped query log data.
            Grouped on query template and query parameters.
        """
        return self._grouped_df_params

    def get_params(self, query):
        """
        Find the parameters associated with a particular query.

        Parameters
        ----------
        query : str
            The query template to look up parameters for.

        Returns
        -------
        params : pd.Series
            The counts of parameters associated with a particular query.
            Unfortunately, due to quirks of the PostgreSQL CSVLOG format,
            the types of parameters are unreliable and may be stringly typed.
        """
        params = self._grouped_df_params.query("query_template == @query")
        return params.droplevel(0).squeeze(axis=1)

    def sample_params(self, query, n, replace=True, weights=True):
        """
        Find a sampling of parameters associated with a particular query.

        Parameters
        ----------
        query : str
            The query template to look up parameters for.
        n : int
            The number of parameter vectors to sample.
        replace : bool
            True if the sampling should be done with replacement.
        weights : bool
            True if the sampling should use the counts as weights.
            False if the sampling should be equal probability weighting.

        Returns
        -------
        params : np.ndarray
            Sample of the parameters associated with a particular query.
        """
        params = self.get_params(query)
        weight_vec = params if weights else None
        sample = params.sample(n, replace=replace, weights=weight_vec)
        return sample.index.to_numpy()

    @staticmethod
    def substitute_params(query_template, params):
        assert type(query_template) == str
        query = query_template
        keys = [f"${i}" for i in range(1, len(params) + 1)]
        for k, v in reversed(list(zip(keys, params))):
            # The reversing is crucial! Note that $1 is a prefix of $10.
            query = query.replace(k, v)
        return query

    @staticmethod
    def _read_csv(csvlog, log_columns):
        """
        Read a PostgreSQL CSVLOG file into a pandas DataFrame.

        Parameters
        ----------
        csvlog : str
            Path to a CSVLOG file generated by PostgreSQL.
        log_columns : List[str]
            List of columns in the csv log.

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the relevant columns for query forecasting.
        """
        # This function must have a separate non-local binding from _read_df
        # so that it can be pickled for multiprocessing purposes.
        return pd.read_csv(
            csvlog,
            names=log_columns,
            parse_dates=["log_time", "session_start_time"],
            usecols=[
                "log_time",
                "session_start_time",
                "command_tag",
                "message",
                "detail",
            ],
            header=None,
            index_col=False,
        )

    @staticmethod
    def _extract_query(message_series):
        """
        Extract SQL queries from the CSVLOG's message column.

        Parameters
        ----------
        message_series : pd.Series
            A series corresponding to the message column of a CSVLOG file.

        Returns
        -------
        query : pd.Series
            A str-typed series containing the queries from the log.
        """
        simple = r"statement: ((?:DELETE|INSERT|SELECT|UPDATE).*)"
        extended = r"execute .+: ((?:DELETE|INSERT|SELECT|UPDATE).*)"
        regex = f"(?:{simple})|(?:{extended})"
        query = message_series.str.extract(regex, flags=re.IGNORECASE)
        # Combine the capture groups for simple and extended query protocol.
        query = query[0].fillna(query[1])
        query.fillna("", inplace=True)
        return query.astype(str)

    @staticmethod
    def _extract_params(detail_series):
        """
        Extract SQL parameters from the CSVLOG's detail column.
        If there are no such parameters, an empty {} is returned.

        Parameters
        ----------
        detail_series : pd.Series
            A series corresponding to the detail column of a CSVLOG file.

        Returns
        -------
        params : pd.Series
            A dict-typed series containing the parameters from the log.
        """

        def extract(detail):
            detail = str(detail)
            prefix = "parameters: "
            idx = detail.find(prefix)
            if idx == -1:
                return {}
            parameter_list = detail[idx + len(prefix):]
            params = {}
            for pstr in parameter_list.split(", "):
                pnum, pval = pstr.split(" = ")
                assert pnum.startswith("$")
                assert pnum[1:].isdigit()
                params[pnum] = pval
            return params

        return detail_series.parallel_apply(extract)

    @staticmethod
    def _substitute_params(df, query_col, params_col):
        """
        Substitute parameters into the query, wherever possible.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe of query log data.
        query_col : str
            Name of the query column produced by _extract_query.
        params_col : str
            Name of the parameter column produced by _extract_params.
        Returns
        -------
        query_subst : pd.Series
            A str-typed series containing the query with parameters inlined.
        """

        def substitute(query, params):
            # Consider '$2' -> "abc'def'ghi".
            # This necessitates the use of a SQL-aware substitution,
            # even if this is much slower than naive string substitution.
            new_sql, last_end = [], 0
            for token in pglast.parser.scan(query):
                token_str = str(query[token.start: token.end + 1])
                if token.start > last_end:
                    new_sql.append(" ")
                if token.name == "PARAM":
                    assert token_str.startswith("$")
                    assert token_str[1:].isdigit()
                    new_sql.append(params[token_str])
                else:
                    new_sql.append(token_str)
                last_end = token.end + 1
            new_sql = "".join(new_sql)
            return new_sql

        def subst(row):
            return substitute(row[query_col], row[params_col])

        return df.parallel_apply(subst, axis=1)

    @staticmethod
    def _parse(query_series):
        """
        Parse the SQL query to extract (prepared queries, parameters).

        Parameters
        ----------
        query_series : pd.Series
            SQL queries with the parameters inlined.

        Returns
        -------
        queries_and_params : pd.Series
            A series containing tuples of (prepared SQL query, parameters).
        """

        def parse(sql):
            new_sql, params, last_end = [], [], 0
            for token in pglast.parser.scan(sql):
                token_str = str(sql[token.start: token.end + 1])
                if token.start > last_end:
                    new_sql.append(" ")
                if token.name in ["ICONST", "FCONST", "SCONST"]:
                    # Integer, float, or string constant.
                    new_sql.append("$" + str(len(params) + 1))
                    params.append(token_str)
                else:
                    new_sql.append(token_str)
                last_end = token.end + 1
            new_sql = "".join(new_sql)
            return new_sql, tuple(params)

        return query_series.parallel_apply(parse)

    def _from_csvlogs(self, workload_csv_path, log_columns, store_query_subst=False):
        """
        Glue code for initializing the Preprocessor from CSVLOGs.

        Parameters
        ----------
        csvlogs : List[str]
            List of PostgreSQL CSVLOG files.
        log_columns : List[str]
            List of columns in the csv log.
        store_query_subst: bool
            True if the "query_subst" column should be stored.

        Returns
        -------
        df : pd.DataFrame
            A dataframe representing the query log.
        """
        time_end, time_start = None, time.perf_counter()

        def clock(label):
            nonlocal time_end, time_start
            time_end = time.perf_counter()
            print("\r{}: {:.2f} s".format(label, time_end - time_start))
            time_start = time_end

        df = self._read_csv(workload_csv_path, log_columns)
        clock("Read dataframe")

        print("Extract queries: ", end="", flush=True)
        df["query_raw"] = self._extract_query(df["message"])
        df.drop(columns=["message"], inplace=True)
        clock("Extract queries")

        print("Extract parameters: ", end="", flush=True)
        df["params"] = self._extract_params(df["detail"])
        df.drop(columns=["detail"], inplace=True)
        clock("Extract parameters")

        print("Substitute parameters into query: ", end="", flush=True)
        df["query_subst"] = self._substitute_params(df, "query_raw", "params")
        df.drop(columns=["query_raw", "params"], inplace=True)
        clock("Substitute parameters into query")

        print("Parse query: ", end="", flush=True)
        parsed = self._parse(df["query_subst"])
        df[["query_template", "query_params"]] = pd.DataFrame(
            parsed.tolist(), index=df.index)
        clock("Parse query")

        # Only keep the relevant columns to optimize for storage, unless otherwise specified.
        stored_columns = ["log_time", "query_template", "query_params"]
        if store_query_subst:
            stored_columns.append("query_subst")
        return df[stored_columns]

    def _build_query_object(self, index: int, query_template: str):
        if "pg_" in query_template:
            return None

        try:
            p = Parser(query_template)
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
                    if column in self._db_schema[table_name]:
                        added = True
                        column_objects.append(
                            self.columns[table_name][column])
                        break

                # Cannot find a corresponding table for this column
                if not added:
                    logging.debug("Invalid Query: " + query_template)
                    return None

        query = Query(index, query_template+";", column_objects)
        return query

    @staticmethod
    def print_workload(workload: Workload):
        for query in workload.queries[:100:5]:
            print(
                f"Q{query.nr}, columns={query.columns}, template={query.text[:25]}\n")

    def _build_table_object(self):
        # Construct table object with columns
        for table_name, column_names in self._db_schema.items():
            table_obj = Table(table_name)
            self.tables[table_name] = table_obj
            self.columns[table_name] = {}
            for column_name in column_names:
                column_obj = Column(column_name)
                column_obj.table = table_name
                table_obj.add_column(column_obj)
                self.columns[table_name][column_name] = column_obj

    def generate_sample_workload(self):
        """Generate a sample that is representative of the entire workload. Count
        of each query is scaled such that sum of occurances of all queries equals
        'sample_size'. Since each query can have many possible parameters, parameters
        are also sampled based on a weight.

        Returns:
            Workload: sampled workload object
        """

        # Should produce 'sample_size' number of queries, weighted by their occurences
        queries = []
        for i, (query_template, count) in enumerate(self._scaled_query_count.iteritems()):
            # weights=True ensures that the parameters are sampled according to their occurences
            params = self.sample_params(
                query_template, n=count, replace=True, weights=True)
            for param in params:
                query_text = self.substitute_params(query_template, param)
                query = self._build_query_object(
                    i, query_text)
                if query is not None:
                    queries.append(query)
                else:
                    # Query is not useful (ie. does not have a where clause), skip this query
                    break
        self._workload = Workload(queries)

        # self.print_workload(self._workload)
        return self._workload

    def __init__(self, workload_csv_path, sample_size=100, store_query_subst=False):
        """
        Args:
            sample_size (int, optional): sample 100 queries from total queries, weighted based on their occurances. Defaults to 100.
        """
        log_columns = self._PG_LOG_COLUMNS

        self.tables = {}
        self.columns = {}

        # TODO: Add rest of the schemas here
        self._db_schema = self._EPINIONS_SCHEMA
        if "epinions" in workload_csv_path:
            self._db_schema = self._EPINIONS_SCHEMA

        self._build_table_object()

        print(f"Preprocessing CSV logs in: {workload_csv_path}")
        df = self._from_csvlogs(
            workload_csv_path, log_columns, store_query_subst=store_query_subst)

        df.set_index("log_time", inplace=True)

        # Grouping queries by template-parameters count.
        gbp = df.groupby(["query_template", "query_params"]).size()
        grouped_by_params = pd.DataFrame(gbp, columns=["count"])
        # Remove unrelated queries and empty queries
        grouped_by_params = grouped_by_params[~grouped_by_params.index.isin([
                                                                            ("", ())])]
        grouped_by_params = grouped_by_params[~grouped_by_params.index.get_level_values(
            0).str.contains('pg_', case=False)]

        # Sample a subset of query to perform index selection
        query_count = grouped_by_params.groupby(
            "query_template")["count"].sum()
        scaled_query_count = (query_count * sample_size /
                              sum(query_count)).apply(np.ceil).astype(int)

        self._df = df
        self._grouped_df_params = grouped_by_params
        self._scaled_query_count = scaled_query_count

# def task_project1():
#     return {
#         # A list of actions. This can be bash or Python callables.
#         "actions": [
#             'echo "Faking action generation."',
#             'echo "SELECT 1;" > actions.sql',
#             'echo "SELECT 2;" >> actions.sql',
#             "echo '{\"VACUUM\": true}' > config.json",
#         ],
#         # Always rerun this task.
#         "uptodate": [False],
#     }


# def task_project1():
#     return {
#         # A list of actions. This can be bash or Python callables.
#         "actions": ['echo "Generating sql commands for t1_epinions_1.sql"',
#                     'echo "CREATE INDEX idx_review_rating ON review(rating);" > t1_epinions_1.sql',
#                     'echo "CREATE INDEX idx_review_i_id ON review(i_id);" >> t1_epinions_1.sql',
#                     'echo "CREATE INDEX idx_trust_source_u_id ON trust(source_u_id);" >> t1_epinions_1.sql',
#                     'echo "CREATE INDEX idx_review_u_id ON review(u_id);" >> t1_epinions_1.sql',
#                     'echo "CREATE INDEX idx_trust_target_u_id ON trust(target_u_id);" >> t1_epinions_1.sql'],
#         # Always rerun this task.
#         "uptodate": [False],
#     }

from selection.project1_index_selection import P1IndexSelection


def project1_setup():
    pass


def task_project1():
    """
    Generate actions.
    """

    # TODO: Install HypoPg
    # sudo apt-get install postgresql-14-hypopg
    # CREATE EXTENSION hypopg;
    # Install dexter

    def run_index_selector(workload_csv, timeout):
        print(f"dodo received workload CSV: {workload_csv}")
        print(f"dodo received timeout: {timeout}")
        ids = P1IndexSelection(workload_csv, None,
                               disable_output_files=True)
        ids.run()

    return {
        "actions": [
            'echo "Faking action generation."',
            run_index_selector,
            # 'echo "SELECT 1;" > actions.sql',
            # 'echo "SELECT 2;" >> actions.sql',
            # 'echo \'{"VACUUM": true}\' > config.json',
        ],
        "uptodate": [False],
        "verbosity": 2,
        "params": [
            {
                "name": "workload_csv",
                "long": "workload_csv",
                "help": "The PostgreSQL workload to optimize for.",
                "default": None,
            },
            {
                "name": "timeout",
                "long": "timeout",
                "help": "The time allowed for execution before this dodo task will be killed.",
                "default": None,
            },
        ],
    }


# PGPASSWORD=project1pass psql --host=localhost --username=project1user --dbname=project1db --file="./t1_epinions_1.sql"

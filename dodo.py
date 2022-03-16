

def task_project1_setup():
    return {
        "actions": [
            'sh install.sh'
        ],
        "verbosity": 2,
    }


def task_project1():
    """
    Generate actions.
    """
    def run_index_selector(workload_csv):
        from selection.project1_index_selection import P1IndexSelection

        print(f"dodo file receive file {workload_csv}...")
        index_selection = P1IndexSelection(workload_csv, data_collection=True, log_level=None,
                                           disable_output_files=True)
        index_selection.run()

    return {
        "actions": [
            'echo "Faking action generation."',
            run_index_selector,
            'echo \'{"VACUUM": true}\' > config.json',
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

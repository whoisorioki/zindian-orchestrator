import duckdb
import os

os.makedirs("reports", exist_ok=True)

con = duckdb.connect("reports/experiments.db")
con.execute("CREATE SEQUENCE IF NOT EXISTS experiments_id_seq")
con.execute("CREATE SEQUENCE IF NOT EXISTS submissions_id_seq")


con.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id       INTEGER PRIMARY KEY DEFAULT nextval('experiments_id_seq'),
        branch_name         VARCHAR NOT NULL,
        oof_rmse            FLOAT,
        feature_count       INTEGER,
        calibration_method  VARCHAR,
        gate_result         VARCHAR,
        gate_reason         VARCHAR,
        md5_target_hash     VARCHAR,
        dag_phase           VARCHAR,
        notes               VARCHAR,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

con.execute("""
    CREATE TABLE IF NOT EXISTS submissions (
        submission_id       INTEGER PRIMARY KEY DEFAULT nextval('submissions_id_seq'),
        experiment_id       INTEGER,
        branch_name         VARCHAR NOT NULL,
        submission_rank     INTEGER,
        public_score        FLOAT,
        private_score       FLOAT,
        my_rank             INTEGER,
        selected_for_final  BOOLEAN DEFAULT FALSE,
        selection_rationale VARCHAR,
        comment             VARCHAR,
        submitted_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    )
""")

print("DuckDB ledger initialized at reports/experiments.db")
print("Tables: experiments, submissions")
con.close()

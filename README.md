# Databricks churn example

Databricks pilot use case

This repo is intended to demonstrate an model training within MLOps workflow on Databricks, where a model is deployed along with its ancillary pipelines to a specified (currently single) Databricks workspace.

Note: project code is masked to remove sensitive data.
Each pipeline is deployed as a [Databricks job](https://docs.databricks.com/data-engineering/jobs/jobs.html), where these jobs are deployed to a Databricks workspace using Databricks Labs' [`dbx`](https://dbx.readthedocs.io/en/latest/index.html) tool. 


## Pipelines

The following pipelines currently defined within the package are:
- `model-train`
    - Trains a scikit-learn Random Forest model, tracking parameters, metrics and model artifacts to MLflow.
- `integration-test`
    - Sample integration test placeholder.

## Setup local development environment (WIP)

1. Setup python virtual environment
2. Configure databricks CLI environment configs to match project dbx config

## Run interactively on select cluster

```
dbx execute --cluster-id= --job=churn --no-rebuild --no-package --environment=nonprod
```

## Unit test

WIP

## Integration test (as run-submit)

```
dbx deploy --files-only --job=churn --environment=nonprod
dbx launch --job=churn --as-run-submit --environment=nonprod --trace
```


## References

DBX documentation
https://dbx.readthedocs.io/en/latest/quickstart.html


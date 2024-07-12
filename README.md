How to run:
    1. Main model:
        Main model starts through the main.ipynb. You can choose to run all cells or run them individually (from top down)
        You can decide if all models and hyperparameters get trained, or if only the most effective model by changing the config.json.
        "test_all_models": false, is standard, by changing to true, all models get trained.
        The Data needed gets installed dynamically and will be saved as individual csvs for further analysis

    2. Exploratory data analysis:
        Exploratory data analysis has its own .ipynb file and can you can start it with run all or individually, same as before.
        The historical_data needed is provided in the repo.

In the metrics.csv you can find the MSE and MAE values of the last run.
In the result_with_timestamp.csv you can find the prediction for the next 24 hours.
The requirements.txt file contains all the installed packages with the corresponding versions

In the documentation.pdf you can find a much more detailed summary of the project.
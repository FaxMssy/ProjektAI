from zenml import step
from sklearn.metrics import mean_squared_error,mean_absolute_error
from typing_extensions import Annotated
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from typing import Tuple

@step
def evaluate_model(model: RandomForestRegressor, X_test:pd.DataFrame,y_test:pd.Series)-> Tuple[Annotated[float,"mse"],Annotated[float,"mae"]]:
    """
    Evaluate the performance of the best model on the test data.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)

    # Print scalar variables directly
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    metrics = pd.DataFrame({
    'Metric': ['MSE', 'MAE'],
    'Value': [mse, mae]
    })
    metrics.to_csv('data/metrics.csv', index=False)

    return mse,mae
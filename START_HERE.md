# START HERE – Your first steps

This project has been prepared using the Kaggle ZIP you uploaded. It includes the extracted competition data and ready-to-run Python scripts.

## Run order on Windows

1. Extract this project ZIP to a normal folder, for example `C:\ML\Kaggle_House_Price_ML_Project`.
2. Open that folder in VS Code.
3. Double-click `install_windows.bat` (or run it in the VS Code terminal).
4. Run `run_eda.bat` and inspect the EDA reports in `outputs/`.
5. Run `train_models.bat`. It compares the included algorithms, evaluates them and creates a Kaggle submission.
6. Run `run_app.bat` to launch the interactive prediction web app.

You can instead run commands manually; see `README.md`.

## Algorithms pre-configured

- Dummy Median Baseline
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

The model is selected by lowest mean 5-fold cross-validation RMSLE on the training portion. A separate 20% holdout validation set reports RMSLE, MAE, RMSE and R². The final selected pipeline is then fitted to all labelled rows and predicts the Kaggle `test.csv` file.

## Existing example run

A training run was tested during project preparation. It selected Extra Trees by mean cross-validation RMSLE. Its holdout metrics are in `outputs/validation_metrics.json`; the full model comparison is in `outputs/model_comparison.csv`. Your own run may produce slightly different results depending on library versions and environment.

The included trained model is an example artifact. Running `train.py` replaces it with your newly trained model.

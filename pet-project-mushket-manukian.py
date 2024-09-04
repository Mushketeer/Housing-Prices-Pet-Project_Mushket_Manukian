{
  "metadata": {
    "kernelspec": {
      "name": "python",
      "display_name": "Python (Pyodide)",
      "language": "python"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "python",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8"
    },
    "kaggle": {
      "accelerator": "none",
      "dataSources": [
        {
          "databundleVersionId": 111096,
          "sourceId": 10211,
          "sourceType": "competition"
        }
      ],
      "dockerImageVersionId": 30761,
      "isGpuEnabled": false,
      "isInternetEnabled": true,
      "language": "python",
      "sourceType": "notebook"
    },
    "papermill": {
      "default_parameters": {},
      "duration": 10.498407,
      "end_time": "2024-09-04T15:11:58.451773",
      "environment_variables": {},
      "exception": null,
      "input_path": "__notebook__.ipynb",
      "output_path": "__notebook__.ipynb",
      "parameters": {},
      "start_time": "2024-09-04T15:11:47.953366",
      "version": "2.6.0"
    }
  },
  "nbformat_minor": 5,
  "nbformat": 4,
  "cells": [
    {
      "id": "66404923",
      "cell_type": "code",
      "source": "import pandas as pd\nfrom sklearn.preprocessing import OrdinalEncoder\nfrom sklearn.model_selection import train_test_split\n\n# Чтение данных\nX = pd.read_csv('train.csv', index_col='Id')\nX_test_full = pd.read_csv('test.csv', index_col='Id')\n\n# Удаление строк с пропущенными значениями в целевой переменной\nX.dropna(axis=0, subset=['SalePrice'], inplace=True)\ny = X.SalePrice              \nX.drop(['SalePrice'], axis=1, inplace=True)\n\n# Разделение данных на обучающий и валидационный наборы\nX_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)\n\n# Определение столбцов с низкой и высокой кардинальностью\nlow_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == \"object\"]\nhigh_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() > 10 and X_train_full[cname].dtype == \"object\"]\n\n# Определение числовых столбцов\nnumeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]\n\n# Выбор столбцов для обработки\ncols_for_OH_encoder = low_cardinality_cols + numeric_cols\nX_train_for_OH_encoder = X_train_full[cols_for_OH_encoder].copy()\nX_valid_for_OH_encoder = X_valid_full[cols_for_OH_encoder].copy()\nX_test_for_OH_encoder = X_test_full[cols_for_OH_encoder].copy()\n\n# One-Hot Encoding\nOne_Hot_encoded_X_train = pd.get_dummies(X_train_for_OH_encoder)\nOne_Hot_encoded_X_valid = pd.get_dummies(X_valid_for_OH_encoder)\nOne_Hot_encoded_X_test = pd.get_dummies(X_test_for_OH_encoder)\n\n# Выравнивание One-Hot Encoded данных\nOne_Hot_encoded_X_train, One_Hot_encoded_X_valid = One_Hot_encoded_X_train.align(One_Hot_encoded_X_valid, join='left', axis=1)\nOne_Hot_encoded_X_train, One_Hot_encoded_X_test = One_Hot_encoded_X_train.align(One_Hot_encoded_X_test, join='left', axis=1)\n\n# Обработка Ordinal Encoding\n# Создание и обучение OrdinalEncoder\nX_train_for_Ord_encoder = X_train_full[high_cardinality_cols].copy()\nX_valid_for_Ord_encoder = X_valid_full[high_cardinality_cols].copy()\nX_test_for_Ord_encoder = X_test_full[high_cardinality_cols].copy()\n\n# Создание OrdinalEncoder и обучение на обучающем наборе\nordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)\nX_train_for_Ord_encoder[high_cardinality_cols] = ordinal_encoder.fit_transform(X_train_for_Ord_encoder[high_cardinality_cols])\n# Применение трансформации к валидационному и тестовому наборам данных\nX_valid_for_Ord_encoder[high_cardinality_cols] = ordinal_encoder.transform(X_valid_for_Ord_encoder[high_cardinality_cols])\nX_test_for_Ord_encoder[high_cardinality_cols] = ordinal_encoder.transform(X_test_for_Ord_encoder[high_cardinality_cols])\n\n# Объединение обработанных данных в один общий датафрейм\nX_train = pd.concat([One_Hot_encoded_X_train, X_train_for_Ord_encoder], axis=1)\nX_valid = pd.concat([One_Hot_encoded_X_valid, X_valid_for_Ord_encoder], axis=1)\nX_test = pd.concat([One_Hot_encoded_X_test, X_test_for_Ord_encoder], axis=1)",
      "metadata": {
        "papermill": {
          "duration": 2.986018,
          "end_time": "2024-09-04T15:11:54.785669",
          "exception": false,
          "start_time": "2024-09-04T15:11:51.799651",
          "status": "completed"
        },
        "tags": [],
        "trusted": true
      },
      "outputs": [],
      "execution_count": 3
    },
    {
      "id": "124a480c",
      "cell_type": "code",
      "source": "# Импортируем модель машинного обучения XGBoost и Mean Absolute Error\nfrom xgboost import XGBRegressor\nfrom sklearn.metrics import mean_absolute_error\n\n# Создаем и обучаем модель с использованием оптимальных параметров\nmodel = XGBRegressor(n_estimators=1000, learning_rate = 0.05, n_jobs=-1, early_stopping_rounds=10)\nmodel.fit(X_train, y_train, \n          eval_set=[(X_valid, y_valid)],\n          verbose=False)\n\n# Делаем и цениваем прогноз\npredictions = model.predict(X_valid)\nprint(\"Mean Absolute Error: \" + str(mean_absolute_error(predictions, y_valid)))",
      "metadata": {
        "papermill": {
          "duration": 2.782307,
          "end_time": "2024-09-04T15:11:57.575883",
          "exception": false,
          "start_time": "2024-09-04T15:11:54.793576",
          "status": "completed"
        },
        "tags": [],
        "editable": true,
        "slideshow": {
          "slide_type": ""
        },
        "trusted": true
      },
      "outputs": [
        {
          "name": "stdout",
          "text": "Mean Absolute Error: 16330.468990796233\n",
          "output_type": "stream"
        }
      ],
      "execution_count": 4
    },
    {
      "id": "3aa00bc2",
      "cell_type": "code",
      "source": "predictions_test = model.predict(X_test)",
      "metadata": {
        "papermill": {
          "duration": 0.103441,
          "end_time": "2024-09-04T15:11:57.685849",
          "exception": false,
          "start_time": "2024-09-04T15:11:57.582408",
          "status": "completed"
        },
        "tags": [],
        "trusted": true
      },
      "outputs": [],
      "execution_count": 5
    },
    {
      "id": "445bb333",
      "cell_type": "code",
      "source": "output = pd.DataFrame({'Id': X_test.index,\n                       'SalePrice': predictions_test})\noutput.to_csv('submission.csv', index=False)",
      "metadata": {
        "papermill": {
          "duration": 0.023506,
          "end_time": "2024-09-04T15:11:57.714599",
          "exception": false,
          "start_time": "2024-09-04T15:11:57.691093",
          "status": "completed"
        },
        "tags": [],
        "trusted": true
      },
      "outputs": [],
      "execution_count": 6
    }
  ]
}
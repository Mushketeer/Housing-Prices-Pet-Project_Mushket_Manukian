import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Чтение данных
X = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Удаление строк с пропущенными значениями в целевой переменной
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Разделение данных на обучающий и валидационный наборы
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Определение столбцов с низкой и высокой кардинальностью
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
high_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() > 10 and X_train_full[cname].dtype == "object"]

# Определение числовых столбцов
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Выбор столбцов для обработки
cols_for_OH_encoder = low_cardinality_cols + numeric_cols
X_train_for_OH_encoder = X_train_full[cols_for_OH_encoder].copy()
X_valid_for_OH_encoder = X_valid_full[cols_for_OH_encoder].copy()
X_test_for_OH_encoder = X_test_full[cols_for_OH_encoder].copy()

# One-Hot Encoding
One_Hot_encoded_X_train = pd.get_dummies(X_train_for_OH_encoder)
One_Hot_encoded_X_valid = pd.get_dummies(X_valid_for_OH_encoder)
One_Hot_encoded_X_test = pd.get_dummies(X_test_for_OH_encoder)

# Выравнивание One-Hot Encoded данных
One_Hot_encoded_X_train, One_Hot_encoded_X_valid = One_Hot_encoded_X_train.align(One_Hot_encoded_X_valid, join='left', axis=1)
One_Hot_encoded_X_train, One_Hot_encoded_X_test = One_Hot_encoded_X_train.align(One_Hot_encoded_X_test, join='left', axis=1)

# Обработка Ordinal Encoding
# Создание и обучение OrdinalEncoder
X_train_for_Ord_encoder = X_train_full[high_cardinality_cols].copy()
X_valid_for_Ord_encoder = X_valid_full[high_cardinality_cols].copy()
X_test_for_Ord_encoder = X_test_full[high_cardinality_cols].copy()

# Создание OrdinalEncoder и обучение на обучающем наборе
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_for_Ord_encoder[high_cardinality_cols] = ordinal_encoder.fit_transform(X_train_for_Ord_encoder[high_cardinality_cols])
# Применение трансформации к валидационному и тестовому наборам данных
X_valid_for_Ord_encoder[high_cardinality_cols] = ordinal_encoder.transform(X_valid_for_Ord_encoder[high_cardinality_cols])
X_test_for_Ord_encoder[high_cardinality_cols] = ordinal_encoder.transform(X_test_for_Ord_encoder[high_cardinality_cols])

# Объединение обработанных данных в один общий датафрейм
X_train = pd.concat([One_Hot_encoded_X_train, X_train_for_Ord_encoder], axis=1)
X_valid = pd.concat([One_Hot_encoded_X_valid, X_valid_for_Ord_encoder], axis=1)
X_test = pd.concat([One_Hot_encoded_X_test, X_test_for_Ord_encoder], axis=1)

# Импортируем модель машинного обучения XGBoost и Mean Absolute Error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Создаем и обучаем модель с использованием оптимальных параметров
model = XGBRegressor(n_estimators=1000, learning_rate = 0.05, n_jobs=-1, early_stopping_rounds=10)
model.fit(X_train, y_train, 
          eval_set=[(X_valid, y_valid)],
          verbose=False)

# Делаем и оцениваем прогноз
predictions = model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

predictions_test = model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions_test})
output.to_csv('submission.csv', index=False)

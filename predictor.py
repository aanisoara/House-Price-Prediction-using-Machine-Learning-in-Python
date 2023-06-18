def predict_(x_to_predict,x_train,y_train):
    import numpy as np
    import xgboost as xgb
    model = xgb.XGBRegressor(n_estimators = 100, max_depth = 10, seed = 42)
    model.fit(x_train, y_train)
    xgb_y_hat = model.predict(x_to_predict)
    return xgb_y_hat
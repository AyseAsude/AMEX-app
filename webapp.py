import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

most_important_features = ["P_2_min","P_2_last", "P_2_mean", "B_1_last", "D_48_last",
                            "D_44_last", "B_2_last", "B_3_last_lag_sub", "R_1_mean", "P_2_last_lag_div", "B_7_mean",
                            "D_51_last", "D_56_mean", "B_10_last", "R_1_std", "D_44_max", "D_56_max", "B_23_last", "R_2_last", "B_8_mean",
                            "R_27_min", "D_41_last_lag_sub", "B_3_last", "D_45_first", "B_4_last_lag_sub", "D_77_mean"
                            ]

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model("reduced_xgboost.json")


data = {
    "P_2_min":[0.9599609375],
    "P_2_last":[0.9599609375],
    "P_2_mean":[0.9599609375],
    "B_1_last":[0.0090179443359375],
    "D_48_last":[0.195556640625],
    "D_44_last":[0.0063934326171875],
    "B_2_last":[0.81201171875],
    "B_3_last_lag_sub":[0.0],
    "R_1_mean":[0.00830841064453125],
    "P_2_last_lag_div":[1.0],
    "B_7_mean":[0.03790283203125],
    "D_51_last":[0.341796875],
    "D_56_mean":[0.110107421875],
    "B_10_last":[1.013671875],
    "R_1_std": [np.nan],
    "D_44_max":[0.0063934326171875],
    "D_56_max":[0.110107421875],
    "B_23_last":[0.0244903564453125],
    "R_2_last":[0.00908660888671875],
    "B_8_mean":[0.00899505615234375],
    "R_27_min":[np.nan],
    "D_41_last_lag_sub":[0.0],
    "B_3_last":[0.0086669921875],
    "D_45_first":[0.40234375],
    "B_4_last_lag_sub":[0.0],
    "D_77_mean":[0.265625],
}

st.header('Input')
df = pd.DataFrame.from_dict(data, orient='columns').reset_index(drop=True)

def user_input_features():
    P_2_min = st.sidebar.slider('P_2_min', -1.0, 1.0, 0.5)
    B_1_last = st.sidebar.slider('B_1_last', -1.5, 1.5, 0.1)
    D_44_max = st.sidebar.slider('D_44_max', 0.0, 3.5, 0.2)
    B_2_last = st.sidebar.slider('B_2_last', 0.0, 1.0, 0.6)
    return P_2_min, B_1_last, D_44_max, B_2_last


P_2_min, B_1_last, D_44_max, B_2_last = user_input_features()
df["P_2_min"] = P_2_min
df["B_1_last"] = B_1_last
df["D_44_max"] = D_44_max
df["B_2_last"] = B_2_last

st.write(df)


st.header('Prediction')
prediction = xgb_clf.predict_proba(df)[:,1]
prediction

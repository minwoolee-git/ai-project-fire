import os

# Render/CPU 환경에서 TensorFlow 부담 줄이기
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import traceback
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow import keras

# TensorFlow 스레드 최소화
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "sanbul2district-divby100.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fires_model.keras")

app = Flask(__name__)
app.config["SECRET_KEY"] = "hard to guess string"
app.config["WTF_CSRF_ENABLED"] = False
bootstrap = Bootstrap5(app)


class LabForm(FlaskForm):
    longitude = StringField("longitude(1~7)", validators=[DataRequired()])
    latitude = StringField("latitude(1~7)", validators=[DataRequired()])
    month = StringField("month(예: 01-Jan)", validators=[DataRequired()])
    day = StringField("day(예: 00-sun ~ 06-sat, 07-hol)", validators=[DataRequired()])
    avg_temp = StringField("avg_temp", validators=[DataRequired()])
    max_temp = StringField("max_temp", validators=[DataRequired()])
    max_wind_speed = StringField("max_wind_speed", validators=[DataRequired()])
    avg_wind = StringField("avg_wind", validators=[DataRequired()])
    submit = SubmitField("Submit")


def build_pipeline():
    fires = pd.read_csv(CSV_PATH, sep=",")
    fires["burned_area"] = np.log(fires["burned_area"] + 1)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(fires, fires["month"]):
        strat_train_set = fires.loc[train_index].reset_index(drop=True)

    fires_train = strat_train_set.drop(["burned_area"], axis=1)

    num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
    cat_attribs = ["month", "day"]

    num_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    full_pipeline.fit(fires_train)

    # 모델 입력 차원 확인용 샘플
    sample = full_pipeline.transform(fires_train.iloc[[0]])
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    sample = np.asarray(sample, dtype=np.float32)

    return full_pipeline, sample.shape[1]


pipeline, INPUT_DIM = build_pipeline()

# compile=False로 로드해서 추론 부담 줄이기
model = keras.models.load_model(MODEL_PATH, compile=False)

# 워밍업
_dummy = np.zeros((1, INPUT_DIM), dtype=np.float32)
_ = model(_dummy, training=False).numpy()


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    form = LabForm()
    error = None

    if form.validate_on_submit():
        try:
            print("POST /prediction received", flush=True)

            input_df = pd.DataFrame({
                "longitude": [float(form.longitude.data)],
                "latitude": [float(form.latitude.data)],
                "month": [form.month.data.strip()],
                "day": [form.day.data.strip()],
                "avg_temp": [float(form.avg_temp.data)],
                "max_temp": [float(form.max_temp.data)],
                "max_wind_speed": [float(form.max_wind_speed.data)],
                "avg_wind": [float(form.avg_wind.data)],
            })
            print("input_df created", flush=True)

            input_prepared = pipeline.transform(input_df)
            if hasattr(input_prepared, "toarray"):
                input_prepared = input_prepared.toarray()

            input_prepared = np.asarray(input_prepared, dtype=np.float32)
            print(f"transform done: {input_prepared.shape}", flush=True)

            # model.predict 대신 직접 호출
            log_pred = float(model(input_prepared, training=False).numpy().ravel()[0])
            print(f"prediction done: {log_pred}", flush=True)

            pred_area = float(np.exp(log_pred) - 1)
            print(f"converted area: {pred_area}", flush=True)

            return render_template("result.html", res=round(pred_area, 2))

        except Exception as e:
            print("prediction error:", flush=True)
            traceback.print_exc()
            error = f"예측 중 오류가 발생했습니다: {e}"

    elif form.is_submitted():
        print(f"form validation failed: {form.errors}", flush=True)
        error = f"입력값 검증 실패: {form.errors}"

    return render_template("prediction.html", form=form, error=error)


if __name__ == "__main__":
    app.run(debug=True)
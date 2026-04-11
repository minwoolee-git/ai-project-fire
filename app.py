import os
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

from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "sanbul2district-divby100.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fires_model.keras")

app = Flask(__name__)
app.config["SECRET_KEY"] = "hard to guess string"
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
    return full_pipeline


pipeline = build_pipeline()
model = keras.models.load_model(MODEL_PATH)


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

            input_prepared = pipeline.transform(input_df)
            if not isinstance(input_prepared, np.ndarray):
                input_prepared = input_prepared.toarray()

            log_pred = float(model.predict(input_prepared, verbose=0)[0][0])
            pred_area = float(np.exp(log_pred) - 1)

            return render_template("result.html", res=round(pred_area, 2))

        except Exception as e:
            error = f"입력값을 다시 확인해 주세요. ({e})"

    return render_template("prediction.html", form=form, error=error)


if __name__ == "__main__":
    app.run(debug=True)
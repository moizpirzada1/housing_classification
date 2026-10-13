import os

from flask import Flask, render_template, request

import predict_utils as pu

app = Flask(__name__)

FIELDS = [
    ("longitude", "Longitude"),
    ("latitude", "Latitude"),
    ("housing_median_age", "Housing median age"),
    ("total_rooms", "Total rooms"),
    ("total_bedrooms", "Total bedrooms"),
    ("population", "Population"),
    ("households", "Households"),
    ("median_income", "Median income (tens of thousands $)"),
]


@app.route("/", methods=["GET", "POST"])
def index():
    values = dict(pu.DEFAULTS)
    values["ocean_proximity"] = pu.OCEAN_OPTIONS[0]
    result, error = None, None
    if request.method == "POST":
        try:
            values = {k: request.form[k] for k, _ in FIELDS}
            values["ocean_proximity"] = request.form["ocean_proximity"]
            result = pu.predict(values)
        except (ValueError, KeyError):
            error = "Please enter a valid number in every field."
    return render_template(
        "index.html", fields=FIELDS, values=values, options=pu.OCEAN_OPTIONS,
        result=result, error=error, accuracy=pu.ACCURACY,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

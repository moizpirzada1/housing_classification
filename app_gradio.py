import gradio as gr

import predict_utils as pu


def run(longitude, latitude, age, rooms, bedrooms, population, households, income, ocean):
    res = pu.predict({
        "longitude": longitude, "latitude": latitude, "housing_median_age": age,
        "total_rooms": rooms, "total_bedrooms": bedrooms, "population": population,
        "households": households, "median_income": income, "ocean_proximity": ocean,
    })
    summary = (
        f"Predicted: {res['label'].upper()} price district "
        f"(High = median house value above ${res['threshold']:,.0f})"
    )
    return summary, {"High": res["p_high"], "Low": res["p_low"]}


d = pu.DEFAULTS
demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Number(d["longitude"], label="Longitude"),
        gr.Number(d["latitude"], label="Latitude"),
        gr.Number(d["housing_median_age"], label="Housing median age"),
        gr.Number(d["total_rooms"], label="Total rooms"),
        gr.Number(d["total_bedrooms"], label="Total bedrooms"),
        gr.Number(d["population"], label="Population"),
        gr.Number(d["households"], label="Households"),
        gr.Number(d["median_income"], label="Median income (tens of thousands $)"),
        gr.Dropdown(pu.OCEAN_OPTIONS, value=pu.OCEAN_OPTIONS[0], label="Ocean proximity"),
    ],
    outputs=[gr.Textbox(label="Result"), gr.Label(label="Class probabilities")],
    title="Housing Price Classifier (Logistic Regression)",
    description=(
        "Predicts whether a California district has a HIGH or LOW median house value. "
        f"Model test accuracy: {pu.ACCURACY:.1%}."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

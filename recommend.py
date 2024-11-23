from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Class MealPlanModel
class MealPlanModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.ingredients_data = {
            "Breakfast": [
                "100g oat flour, 200g bananas, 150g egg whites",
                "150g yogurt, 50g granola, 20g honey",
                "2 slices whole grain bread, 1 avocado, 2 boiled eggs",
                "200ml almond milk, 1 scoop protein powder, 100g blueberries"
            ],
            "Lunch": [
                "200g chicken breast, 150g quinoa, 100g mixed vegetables",
                "150g turkey, 100g sweet potatoes, 150g steamed broccoli",
                "200g tofu, 150g brown rice, 100g stir-fried vegetables",
                "200g beef steak, 150g mashed potatoes, 100g green beans"
            ],
            "Dinner": [
                "200g salmon, 150g brown rice, 100g mixed vegetables",
                "150g grilled shrimp, 150g zucchini noodles, 20g olive oil",
                "200g baked chicken thighs, 150g roasted asparagus, 100g mashed sweet potatoes",
                "200g cod fish, 150g couscous, 100g sautéed spinach"
            ]
        }

    def train(self, data):
        features = ["age", "height", "weight"]  # Removed 'goal'
        target = ["calories", "protein", "carbs", "fat"]

        X = data[features]
        y = data[target]

        numerical_features = ["age", "height", "weight"]

        numerical_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features)
            ]
        )

        self.model = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("regressor", RandomForestRegressor(random_state=42))
        ])

        self.model.fit(X, y)

    def predict_weekly(self, age, height, weight):
        user_input = pd.DataFrame([{
            "age": age,
            "height": height,
            "weight": weight
        }])

        prediction = self.model.predict(user_input)

        base_plan = {
            "calories": int(prediction[0][0]),
            "protein": int(prediction[0][1]),
            "carbs": int(prediction[0][2]),
            "fat": int(prediction[0][3])
        }

        week_plan = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        for day in days:
            daily_variation = np.random.uniform(0.9, 1.1, 4)
            daily_plan = {
                "calories": int(base_plan["calories"] * daily_variation[0]),
                "protein": int(base_plan["protein"] * daily_variation[1]),
                "carbs": int(base_plan["carbs"] * daily_variation[2]),
                "fat": int(base_plan["fat"] * daily_variation[3]),
                "meals": [
                    {
                        "type": "Breakfast",
                        "ingredients": np.random.choice(self.ingredients_data["Breakfast"])
                    },
                    {
                        "type": "Lunch",
                        "ingredients": np.random.choice(self.ingredients_data["Lunch"])
                    },
                    {
                        "type": "Dinner",
                        "ingredients": np.random.choice(self.ingredients_data["Dinner"])
                    }
                ]
            }
            week_plan[day] = daily_plan

        return week_plan

# Train the model
np.random.seed(42)
data = pd.DataFrame({
    "age": np.random.randint(18, 60, 1000),
    "height": np.random.randint(150, 200, 1000),
    "weight": np.random.randint(50, 100, 1000),
    "calories": np.random.randint(1500, 3000, 1000),
    "protein": np.random.randint(50, 200, 1000),
    "carbs": np.random.randint(100, 300, 1000),
    "fat": np.random.randint(30, 100, 1000)
})

meal_plan_model = MealPlanModel()
meal_plan_model.train(data)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    request_data = request.json
    age = request_data["age"]
    height = request_data["height"]
    weight = request_data["weight"]

    weekly_plan = meal_plan_model.predict_weekly(age, height, weight)
    return jsonify(weekly_plan)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

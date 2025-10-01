import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------
# 1️⃣ Setup
# ---------------------------
app = Flask(__name__)
load_dotenv()

# Load model + encoders
MODEL_PATH = os.path.join("models", "booking_model.pkl")
model, le_deposit, le_customer = joblib.load(MODEL_PATH)

# Gemini API client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️ GEMINI_API_KEY not set in .env")
client = genai.Client(api_key=api_key)


# ---------------------------
# 2️⃣ Helper function
# ---------------------------
def generate_explanation_manual(
    prediction,
    hotel, lead_time, arrival_date_month, stays_in_weekend_nights, stays_in_week_nights,
    adults, children, babies,
    meal, country, market_segment, distribution_channel, is_repeated_guest,
    previous_cancellations, previous_bookings_not_canceled, reserved_room_type,
    assigned_room_type, booking_changes, deposit_type, adr,
    customer_type, required_car_parking_spaces, total_of_special_requests
):
    return f"""
    The model predicted that this booking will be {'Cancelled' if prediction==1 else 'Not Cancelled'}.

    Booking Details:
    - Hotel: {hotel}
    - Lead Time: {lead_time}
    - Arrival Month: {arrival_date_month}
    - Weekend Nights: {stays_in_weekend_nights}
    - Week Nights: {stays_in_week_nights}
    - Adults: {adults}
    - Children: {children}
    - Babies: {babies}

    Financial Details:
    - Meal: {meal}
    - Country: {country}
    - Market Segment: {market_segment}
    - Distribution Channel: {distribution_channel}
    - Is Repeated Guest: {is_repeated_guest}
    - Previous Cancellations: {previous_cancellations}
    - Previous Bookings Not Cancelled: {previous_bookings_not_canceled}
    - Reserved Room Type: {reserved_room_type}
    - Assigned Room Type: {assigned_room_type}
    - Booking Changes: {booking_changes}
    - Deposit Type: {deposit_type}
    - ADR: {adr}

    Other Details:
    - Customer Type: {customer_type}
    - Required Car Parking Spaces: {required_car_parking_spaces}
    - Total Special Requests: {total_of_special_requests}

    Explain why the model might have given this prediction.
    """


# ---------------------------
# 3️⃣ Flask Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html", deposits=le_deposit.classes_, customers=le_customer.classes_)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form  # from HTML form

        # Encode categorical inputs
        deposit_type_enc = le_deposit.transform([data["deposit_type"]])[0]
        customer_type_enc = le_customer.transform([data["customer_type"]])[0]

        # Arrange input for ML
        X_input = np.array([[
            int(data["lead_time"]),
            float(data["adr"]),
            int(data["booking_changes"]),
            int(data["previous_cancellations"]),
            int(data["total_of_special_requests"]),
            deposit_type_enc,
            customer_type_enc
        ]])

        # Prediction
        prediction = int(model.predict(X_input)[0])

        # Build explanation
        explanation_prompt = generate_explanation_manual(
            prediction,
            hotel="N/A",
            lead_time=data["lead_time"],
            arrival_date_month="N/A",
            stays_in_weekend_nights=0,
            stays_in_week_nights=0,
            adults=0,
            children=0,
            babies=0,
            meal="N/A",
            country="N/A",
            market_segment="N/A",
            distribution_channel="N/A",
            is_repeated_guest=0,
            previous_cancellations=data["previous_cancellations"],
            previous_bookings_not_canceled=0,
            reserved_room_type="N/A",
            assigned_room_type="N/A",
            booking_changes=data["booking_changes"],
            deposit_type=data["deposit_type"],
            adr=data["adr"],
            customer_type=data["customer_type"],
            required_car_parking_spaces=0,
            total_of_special_requests=data["total_of_special_requests"]
        )

        # Gemini explanation
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[explanation_prompt],
            config=types.GenerateContentConfig(
                system_instruction="Explain the model prediction in simple language."
            )
        )

        return render_template(
            "index.html",
            deposits=le_deposit.classes_,
            customers=le_customer.classes_,
            prediction="Cancelled" if prediction == 1 else "Not Cancelled",
            explanation=response.text.strip()
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# 4️⃣ Run Flask
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)

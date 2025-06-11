import streamlit as st
import io
import re
import unicodedata
import html
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key="AIzaSyAOhZzjwxbGtYuCKvyic9xvdX9FmaX2xm8")

# Load images
def load_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = load_base64("background.png") 
logo_base64 = load_base64("eato_logo.png")
logo2_base64 = load_base64("eato_logo2.png")
about_illus_base64 = load_base64("about_illustration.png")
loading_base64 = load_base64("loading.gif")

# Set page config
st.set_page_config(page_title="EATO", 
                   page_icon=f"data:image/png;base64,{logo_base64}",
                   layout="centered")

# Hide Streamlit default elements
st.markdown("""
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container { padding: 0 !important; margin: 0 !important; }
    </style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "landing"

# Read current URL query parameters
query_params = st.query_params
if "page" in query_params and query_params["page"]:
    st.session_state.page = query_params["page"]

# -----------------------------
# LANDING PAGE 
# -----------------------------
if st.session_state.page == "landing":
    st.markdown(f"""
        <style>
        html, body {{
            height: 100%;
            margin: 0;
            overflow: hidden;
        }}
        .hero {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: url("data:image/jpeg;base64,{bg_base64}") no-repeat center center;
            background-size: cover;
            font-family: 'Segoe UI', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 1rem;
            pointer-events: none;
        }}
        .content {{
            color: white;
            max-width: 700px;
        }}
        .content h1 {{
            font-size: 4rem;
            font-weight: 700;
            margin-bottom:0.1rem;
        }}
        .content p {{
            font-size: 1.5rem;
            font-weight: 150;
            margin-bottom: 2rem;
        }}
        .stButton > button {{
            background: white;
            color: #28a745;
            border: none;
            padding: 1rem 2.5rem;
            font-size: 1.25rem;
            font-weight: 900 !important;
            font-family: "Arial Black", sans-serif !important; 
            border-radius: 50px;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease 0s;
            cursor: pointer;
        }}
        .stButton > button:hover {{
            background: #f0f0f0;
            color: #1c7c36;
            transform: translateY(-10px);
        }}
        .logo {{
            position: absolute;
            top: 20px;
            left: 20px;
        }}
        .logo img {{
            height: 60px;
        }}
        </style>

        <div class="hero">
            <div class="logo">
                <img src="data:image/png;base64,{logo2_base64}" />
            </div>
            <div class="content">
                <h1>A healthy life starts<br>with what you eat</h1>
                <p>Get personalized diet plans and recipe suggestions<br>tailored to your lifestyle.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Centered button to start
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        if st.button("➜   GET STARTED NOW"):
            st.session_state.page = "form"
            st.query_params.clear()  # Clear query params from URL
            st.rerun()

# -----------------------------
# FORM PAGE
# -----------------------------
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if st.session_state.page == "form":
    # Navbar
    st.markdown(f"""
        <style>
        .navbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 1rem;
            background-color: white;
            border-bottom: 1px solid #eee;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            font-family: 'Segoe UI', sans-serif;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
        }}
        .navbar-left {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .navbar-left img {{
            height: 60px;
        }}
        .navbar-left h1 {{
            font-size: 18px;
            margin: 0;
            font-weight: 700;
            color: #222;
        }}
        .navbar-right {{
            display: flex;
            gap: 24px;
            font-size: 14px;
        }}
        .navbar-right a {{
            color: #333;
            text-decoration: none;
            font-weight: 500;
        }}
        .navbar-right a:hover {{
            color: #1abc9c;
        }}
        .block-container {{
            padding-top: 5rem !important;
        }}
        </style>

        <div class="navbar">
            <div class="navbar-left">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo">
                <h1>Personalized Diet Recommender</h1>
            </div>
            <div class="navbar-right">
                <a href="?page=landing">Home</a>
                <a href="?page=faqs">FAQs</a>
                <a href="?page=about">About</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Form content
    with st.form("user_input_form"):
        st.markdown("### 👤 Personal Info")
        col1, col2 = st.columns(2)
        with col1:
            # Using text input with placeholder so these fields are empty by default
            height = st.number_input("Height (in cm)", min_value=50, max_value=250, value=None, placeholder="Enter your height")
            age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter your age")

        with col2:
            weight = st.number_input("Weight (in kg)", min_value=20, max_value=200, value=None, placeholder="Enter your weight")
            gender = st.radio("Gender", ["Male", "Female"])

        st.markdown("### 🧘 Activity Level")
        activity_level = st.selectbox("Select your activity level", [
            "--- Choose a activity level ---",
            "Sedentary (little to no exercise)",
            "Lightly active (light exercise 1-3 days/week)",
            "Moderately active (moderate exercise 3-5 days/week)",
            "Very active (hard exercise 6-7 days/week)",
            "Super active (very intense exercise or physical job)"
        ])

        st.markdown("### 🎯 Health Goal")
        health_goal = st.selectbox("Select your health goal", [
            "--- Choose a health goal ---", "Weight Loss", "Maintain Weight", "Gain Muscle"
        ])
        
        st.markdown("### ⚠️ Allergies")
        allergen_options = [
            "Milk Allergy / Lactose Intolerance", "Oral Allergy Syndrome", "Legume Allergy", "Fish Allergy", 
            "Crustacean Allergy", "Stone Fruit Allergy", "Fruit Allergy", "Animal Allergy", 
            "Alpha-gal Syndrome", "Salicylate Sensitivity", "Sugar Allergy / Intolerance", 
            "Histamine Allergy", "Insulin Allergy", "Nut Allergy", "Seed Allergy", "Pollen Allergy", 
            "Gluten Allergy", "Corn Allergy", "Mushroom Allergy", "LTP Allergy", "Honey Allergy", 
            "Shellfish Allergy", "Peanut Allergy", "Tannin Allergy", "Soy Allergy", "Pepper Allergy", 
            "Lactose Allergy", "Rice Allergy", "Rose Allergy", "Aquagenic Urticaria", "Orchidaceae Allergy", 
            "Banana Allergy", "Broccoli Allergy", "Beef Allergy", "Ragweed Allergy", "Thyroid"
        ]

        allergies = st.multiselect("Select known allergies", options=allergen_options)
        # Zigzag option right after allergy selection

        if "use_zigzag" not in st.session_state:
            st.session_state.use_zigzag = True

        st.markdown("")
        use_zigzag = st.checkbox("Apply weekly calorie zigzag", value=st.session_state.get("use_zigzag", True))
        st.session_state.use_zigzag = use_zigzag
        st.markdown("")

        # Clear previous zigzag day selection if toggle is off
        if not st.session_state.use_zigzag and "selected_day" in st.session_state:
            del st.session_state.selected_day

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.form_submit_button("🍽️ Get Recipe Recommendations"):
                # Reset saved recipes and used IDs when form is submitted
                for day in range(7):
                    for meal in range(3):
                        st.session_state.pop(f"zigzag_recipe_day{day}_meal{meal}", None)

                st.session_state.zigzag_used_recipes = set()
                st.session_state.zigzag_plan_generated = False  # <=== VERY IMPORTANT!!
                st.session_state.selected_day = 0  # Reset back to Day 1
                st.session_state.form_submitted = True
                st.rerun()

    def show_macro_comparison(recipe_macros, user_macros, title="Macronutrient Comparison"):
        labels = ['Carbs', 'Protein', 'Fat']
        recipe_vals = [recipe_macros['carbs'], recipe_macros['protein'], recipe_macros['fat']]
        user_vals = [user_macros['carbs'], user_macros['protein'], user_macros['fat']]

        x = range(len(labels))
        width = 0.25

        fig = plt.figure(figsize=(2.5, 1.5), dpi=120)
        ax = fig.add_subplot(111)
        ax.bar([i - width/2 for i in x], user_vals, width, label='Needed', color='#1f77b4')
        ax.bar([i + width/2 for i in x], recipe_vals, width, label='In Recipes', color='#ff7f0e')

        ax.set_ylabel('Grams', fontsize=5)
        ax.set_title(title, fontsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5)
        ax.tick_params(axis='y', labelsize=5)
        ax.legend(fontsize=3)

        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)

    def get_health_advice(age, weight, height, goal, activity_level, plan_df=None):        
        prompt = ""

        if plan_df is not None:
            prompt += """
    You are a warm, supportive, and knowledgeable health advisor.
    First, based on the following 7-day meal plan, provide some insights:
    Here is the user's meal plan:
    """
            for idx, row in plan_df.iterrows():
                prompt += f"- {row['Day']} {row['Meal']}: {row['Recipe Name']} ({row['Calories']} kcal)\n"
                prompt += f"    Ingredients: {row['Ingredients']}\n"
                prompt += f"    Macros: {row['Carbs (g)']}g Carbs, {row['Protein (g)']}g Protein, {row['Fat (g)']}g Fat\n\n"

            prompt += """
    Tasks:
    - Write a short, positive overview about the provided meal plan and how it help the user in 3–5 sentences.
    - Create a clear, well-formatted grocery shopping list as a table.
        - Only list the main ingredients (no need for quantities).
        - Group ingredients into logical categories: Vegetables, Fruits, Proteins, Grains, Dairy, Others.
        - Format the table properly using Markdown if possible.
        - Keep it clean, professional, and easy to read.
    - Suggest 3–5 practical meal prep tips to make the week easier.
    - End this section with a short motivational message about staying on track.

    Use friendly, supportive, and encouraging language. 🌟
    """

        # Now the second part: standard health advice
        prompt += f"""

    Next, based on the user's personal profile:

    - Age: {age} years old
    - Weight: {weight} kg
    - Height: {height} cm
    - Health Goal: {goal}
    - Activity Level: {activity_level}

    Write a personalized health tip in a positive, motivational tone.

    Guidelines:
    - Keep it between 3 to 5 sentences.
    - Use friendly, supportive language (avoid robotic tone).
    - Give practical advice related to exercise, sleep, hydration, and mental well-being.
    - Emphasize consistency over perfection — encourage small daily steps.
    - Remind the user to celebrate small wins.
    - Optionally, end with a motivating tagline like "You've got this!" or "Your journey is worth it!".
    """
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"❌ Failed to generate advice: {e}"
    
    def format_duration(duration_str):
        if not isinstance(duration_str, str):
            return "N/A"

        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?', duration_str)
        if not match:
            return "N/A"

        hours, minutes = match.groups()
        parts = []
        if hours:
            parts.append(f"{int(hours)} hr")
        if minutes:
            parts.append(f"{int(minutes)} min")
        return " ".join(parts) if parts else "N/A"

    def clean_text(text):
        if not isinstance(text, str):
            return text
        # Normalize weird unicode
        text = unicodedata.normalize("NFKD", text)
        # Remove control characters
        text = ''.join(c for c in text if unicodedata.category(c)[0] != "C")
        # Remove broken line continuations (like '1/8\')
        text = text.replace('\\', '')
        # Remove isolated commas
        text = re.sub(r'^[,]+$', '', text, flags=re.MULTILINE)
        # Strip whitespace
        text = text.strip()
        return text

    def create_downloadable_plan_from_session():
        plan_rows = []
        max_steps = 10  # Maximum number of steps to include

        for day in range(7):
            for meal_idx, meal_name in enumerate(["Breakfast", "Lunch", "Dinner"]):
                key = f"zigzag_recipe_day{day}_meal{meal_idx}"
                row = st.session_state.get(key)

                if row is not None:
                    instructions = parse_r_list(row.get("RecipeInstructions", ""))
                    ingredients = parse_r_list(row.get("RecipeIngredientParts", ""))

                    entry = {
                        "Day": f"Day {day+1}",
                        "Meal": meal_name,
                        "Recipe Name": row.get("Name", "N/A"),
                        "Calories": int(row.get("Calories", 0)),
                        "Carbs (g)": row.get("CarbohydrateContent", "N/A"),
                        "Protein (g)": row.get("ProteinContent", "N/A"),
                        "Fat (g)": row.get("FatContent", "N/A"),
                        "Prep Time": format_duration(row.get("PrepTime", "N/A")),
                        "Cook Time": format_duration(row.get("CookTime", "N/A")),
                        "Total Time": format_duration(row.get("TotalTime", "N/A")),
                        "Ingredients": ", ".join(ingredients) if ingredients else "N/A",
                    }

                    # Add Step 1, Step 2, ..., Step 10
                    for i in range(max_steps):
                        step_text = instructions[i] if i < len(instructions) else ""
                        entry[f"Step {i+1}"] = step_text

                    plan_rows.append(entry)

                else:
                    # If no recipe, fill empty row
                    entry = {
                        "Day": f"Day {day+1}",
                        "Meal": meal_name,
                        "Recipe Name": "No recipe available",
                        "Calories": "N/A",
                        "Carbs (g)": "N/A",
                        "Protein (g)": "N/A",
                        "Fat (g)": "N/A",
                        "Prep Time": "N/A",
                        "Cook Time": "N/A",
                        "Total Time": "N/A",
                        "Ingredients": "N/A",
                    }

                    for i in range(max_steps):
                        entry[f"Step {i+1}"] = ""

                    plan_rows.append(entry)

        return pd.DataFrame(plan_rows)


    if st.session_state.form_submitted:
        if (
            None in [height, weight, age] or
            activity_level.startswith("---") or
            health_goal.startswith("---")
        ):
            st.error("🚫 Please complete all fields before submitting.")
        else:
            # BMI Safety Checks 
            bmi = weight / (height / 100) ** 2
            bmi = round(bmi, 1) 
            if bmi < 18.5 and health_goal == "Weight Loss":
                st.warning(f"Healthy BMI range is typically **18.5 - 24.9**. ⚠️ Your BMI is **{bmi}**, which indicates you are underweight. Weight loss may not be recommended. Please consult a healthcare provider.")
            elif bmi > 24.9 and health_goal == "Gain Muscle":
                st.warning(f"Healthy BMI range is typically **18.5 - 24.9**. ⚠️ Your BMI is **{bmi}**, which is in the overweight range. You might want to focus on fat loss first before muscle gain, depending on your goals. Always consult a professional if unsure.")
            # Create placeholder for loading spinner
            loading_placeholder = st.empty()
            # Show loading animation
            with loading_placeholder:
                components.html(f"""
                    <div style="text-align: center; padding: 1rem;">
                        <h2 style="font-family: 'Segoe UI', sans-serif; font-size: 1.6rem; margin-bottom: 1rem;">
                            🍽️ Loading the recipe... Please wait a "yummy" moment.
                        </h2>
                        <img src="data:image/gif;base64,{loading_base64}" width="200">
                    </div>
                """, height=260)

            # 1. Calculate BMR using Mifflin-St Jeor
            s = 5 if gender == "Male" else -161
            bmr = 10 * weight + 6.25 * height - 5 * age + s

            # 2. Get activity multiplier
            activity_multipliers = {
                "Sedentary (little to no exercise)": 1.2,
                "Lightly active (light exercise 1-3 days/week)": 1.375,
                "Moderately active (moderate exercise 3-5 days/week)": 1.55,
                "Very active (hard exercise 6-7 days/week)": 1.725,
                "Super active (very intense exercise or physical job)": 1.9
            }
            tdee = bmr * activity_multipliers.get(activity_level, 1.2)

            # 3. Adjust TDEE and set meal splits & macro ratios &
            if health_goal == "Weight Loss":
                tdee -= 500
                meal_split = [0.45, 0.35, 0.20]
            elif health_goal == "Gain Muscle":
                tdee += 300
                meal_split = [0.30, 0.40, 0.30]
            else:
                meal_split = [0.33, 0.33, 0.33]

            if health_goal and not health_goal.startswith("---"):
                if health_goal == "Maintain Weight":
                    selected_macro = "40% Carbs / 30% Protein / 30% Fat"
                elif health_goal == "Weight Loss":
                    selected_macro = "20% Carbs / 50% Protein / 30% Fat"
                elif health_goal == "Gain Muscle":
                    selected_macro = "40% Carbs / 35% Protein / 25% Fat"
                else:
                    selected_macro = None
            else:
                selected_macro = None
            # Parse macro_vector from user choice
            macro_vector = list(map(int, re.findall(r"\d+", selected_macro)))

            meal_names = ["Breakfast", "Lunch", "Dinner"]
            meal_calories = [round(tdee * pct) for pct in meal_split]

            # 4. Load scaler and KMeans model
            @st.cache_resource
            def load_models_and_data():
                scaler = joblib.load("scaler.pkl")
                kmeans = joblib.load("kmeans_model.pkl")
                df = pd.read_csv("recipes.csv")
                return scaler, kmeans, df

            scaler, kmeans, recipes_df = load_models_and_data()

            macro_scaled = scaler.transform([macro_vector])
            cluster_id = kmeans.predict(macro_scaled)[0]

            # Use the existing 'KMeans_Cluster' column 
            cluster_recipes = recipes_df[recipes_df["KMeans_Cluster"] == cluster_id]

            # 6. Optional: Filter out allergic recipes
            if allergies:
                cluster_recipes = cluster_recipes[~cluster_recipes["AllergyType"].astype(str).str.contains('|'.join(allergies), case=False, na=False)]

            # 7. Define function to get top recipes per meal
            def get_recipes_for_meal(target_kcal, macro_vector, tolerance=25, top_n=1):
                filtered = cluster_recipes[
                    (cluster_recipes["Calories"] >= target_kcal - tolerance) &
                    (cluster_recipes["Calories"] <= target_kcal + tolerance)
                ].copy()

                if filtered.empty:
                    return filtered

                # Calculate macro distance (Euclidean) from user's macro_vector
                def macro_distance(row):
                    return np.linalg.norm([
                        row["Carb_%"] - macro_vector[0],
                        row["Protein_%"] - macro_vector[1],
                        row["Fat_%"] - macro_vector[2]
                    ])

                filtered["macro_score"] = filtered.apply(macro_distance, axis=1)
                return filtered.sort_values("macro_score").head(top_n)

            # 8. Show result to user
            # Clear the loading spinner
            loading_placeholder.empty()
            # Generate Zig-Zag Calorie Plan (7-Day)
            if st.session_state.use_zigzag:
                zigzag_factors = [+0.10, -0.10, +0.05, -0.05, 0, +0.10, -0.10]
                zigzag_days = []
                zigzag_day_names = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

                for factor in zigzag_factors:
                    adjusted = tdee + (tdee * factor)
                    zigzag_days.append(round(adjusted))

                if st.session_state.use_zigzag and not st.session_state.get("zigzag_plan_generated", False):

                    used_recipe_ids = set()

                    for day_idx, factor in enumerate(zigzag_factors):
                        adjusted = tdee + (tdee * factor)
                        daily_meal_calories = [round(adjusted * pct) for pct in meal_split]

                        for meal_idx, kcal in enumerate(daily_meal_calories):
                            matched = get_recipes_for_meal(kcal, macro_vector, top_n=10)
                            available = matched[~matched["RecipeId"].isin(used_recipe_ids)]

                            key = f"zigzag_recipe_day{day_idx}_meal{meal_idx}"

                            if not available.empty:
                                selected = available.sample(1).iloc[0]
                                st.session_state[key] = selected
                                used_recipe_ids.add(selected["RecipeId"])
                            else:
                                st.session_state[key] = None

                    st.session_state.zigzag_plan_generated = True

                st.info(f"""
                #### 🔥 Total Daily Calorie Need (Without Zig-Zag Plan): **{int(tdee)} kcal**

                **Suggested Meal Breakdown:**  
                - 🌞 Breakfast: {meal_calories[0]} kcal  
                - 🌤️ Lunch: {meal_calories[1]} kcal  
                - 🌙 Dinner: {meal_calories[2]} kcal
                """)

                # Day selector UI
                st.markdown("### 🗓️ Your 7-Day Zig-Zag Recipe Plan")
                if "selected_day" not in st.session_state:
                    st.session_state.selected_day = 0
                day_cols = st.columns(7)
                for i in range(7):
                    label = f"🗓️ {zigzag_day_names[i]}"
                    if st.session_state.get("selected_day", 0) == i:
                        label += " ✅"
                    if day_cols[i].button(label):
                        st.session_state.selected_day = i
                        st.rerun()


                selected_day_idx = st.session_state.get("selected_day", 0)
                selected_calories = zigzag_days[selected_day_idx]
                zigzag_pct = round((selected_calories / tdee) * 100)
                # Override daily calories with zigzag day's value
                daily_meal_calories = [round(selected_calories * pct) for pct in meal_split]

                # Track recipes used across the 7-day zigzag
                if "zigzag_used_recipes" not in st.session_state:
                    st.session_state.zigzag_used_recipes = set()

                # Get selected day index
                day_index = st.session_state.get("selected_day", 0)

                st.success(f"""
                🍽️ Showing meals for **{zigzag_day_names[selected_day_idx]}** 

                🔥 Daily Target: **{selected_calories} kcal** (**{zigzag_pct}% TDEE**) 

                **Suggested Meal Breakdown:**  
                - 🌞 Breakfast: {daily_meal_calories[0]} kcal  
                - 🌤️ Lunch: {daily_meal_calories[1]} kcal  
                - 🌙 Dinner: {daily_meal_calories[2]} kcal
                """)

            else:
                # --- Standard Single-Day Plan ---
                st.info(f"""
                #### 🔥 Total Daily Calorie Need: **{int(tdee)} kcal**

                **Suggested Meal Breakdown:**  
                - 🌞 Breakfast: {meal_calories[0]} kcal  
                - 🌤️ Lunch: {meal_calories[1]} kcal  
                - 🌙 Dinner: {meal_calories[2]} kcal
                """)
                daily_meal_calories = meal_calories  


            # Track used recipes to avoid repeats across meals in a day
            used_recipe_ids = set()

            #st.success(f"✅ Assigned to Cluster #{cluster_id}")
            # before looping Breakfast, Lunch, Dinner
            if st.session_state.use_zigzag:
                day_index = st.session_state.get("selected_day", 0)
            else:
                day_index = 0
            # 9. Show recipes
            for i, meal in enumerate(meal_names):
                kcal = daily_meal_calories[i]  # now using zig-zag calories
                matched = get_recipes_for_meal(kcal, macro_vector, top_n=10)
                matched = matched[~matched["RecipeId"].isin(used_recipe_ids)]  # avoid repeated recipes
                key = f"zigzag_recipe_day{day_index}_meal{i}"  # i = 0, 1, 2 for Breakfast, Lunch, Dinner

                available = matched[~matched["RecipeId"].isin(st.session_state.zigzag_used_recipes)]

                if key not in st.session_state:
                    if not available.empty:
                        st.session_state[key] = available.sample(1).iloc[0]
                        st.session_state.zigzag_used_recipes.add(st.session_state[key]["RecipeId"])
                    else:
                        st.session_state[key] = None

                row = st.session_state[key]

                st.markdown(f"""
                <div style="
                    background-color: #f8f8f8;
                    padding: 0.75rem 1.25rem;
                    border-radius: 12px;
                    border: 1px solid #ddd;
                    margin-top: 1rem;
                    margin-bottom: 0.5rem;
                    box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.05);
                ">
                    <h4 style="
                        margin: 0;
                        font-size: 1.8rem;
                        font-weight: 600;
                        color: #333;
                        display: flex;
                        align-items: center;
                    ">
                        🍽️ {clean_text(meal)} Recipes
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                if not matched.empty:
                    if row is not None:

                        # Optional image display
                        # Helper to extract first image from R-style string
                        def extract_first_image(images_string):
                            if not isinstance(images_string, str):
                                return None
                            urls = re.findall(r'"(https?://[^"]+)"', images_string)
                            return urls[0] if urls else None

                        # Helper to parse R-style c("item1", "item2") strings
                        def parse_r_list(r_string):
                            if not isinstance(r_string, str):
                                return []
                            return re.findall(r'"(.*?)"', r_string)

                        # Get image and show
                        img_url = extract_first_image(row.get("Images", ""))
                        if img_url:
                            st.image(img_url, width=400)

                        # Display recipe title, macros, etc.
                        st.markdown(f"""
                        ### 🍲 {clean_text(row['Name'])}

                        **🔥 Calories:** {int(row['Calories'])} kcal  
                        **🆔 Recipe ID:** {row['RecipeId']}  

                        **💪 Macronutrients:**  
                        - 🥖 Carbs: {row['CarbohydrateContent']} g  
                        - 🥩 Protein: {row['ProteinContent']} g  
                        - 🧈 Fat: {row['FatContent']} g  

                        **🕒 Time Info:**  
                        - 🧑‍🍳 Prep Time: {format_duration(row.get('PrepTime', 'N/A'))}  
                        - 🔥 Cook Time: {format_duration(row.get('CookTime', 'N/A'))}  
                        - ⏰ Total Time: {format_duration(row.get('TotalTime', 'N/A'))}  
                        """)
                        #  Display Ingredients with Quantities 
                        ingredients = parse_r_list(row.get("RecipeIngredientParts", ""))
                        quantities = parse_r_list(row.get("RecipeIngredientQuantities", ""))

                        # Ensure the two lists are the same length
                        ingredient_list = []
                        for i, ingredient in enumerate(ingredients):
                            qty = quantities[i] if i < len(quantities) else ""
                            text = f"{clean_text(ingredient)} ({clean_text(qty)})" if qty else clean_text(ingredient)
                            ingredient_list.append(text)

                        if ingredient_list:
                            st.markdown("**🧂 Ingredients:**", unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            half = len(ingredient_list) // 2 + len(ingredient_list) % 2

                            def render_compact_list(items):
                                html = "<ul style='margin: 0; padding-left: 1rem; line-height: 1.6; font-size: 0.95rem;'>"
                                for item in items:
                                    html += f"<li>{clean_text(item)}</li>"
                                html += "</ul>"
                                return html

                            with col1:
                                st.markdown(render_compact_list(ingredient_list[:half]), unsafe_allow_html=True)
                            with col2:
                                st.markdown(render_compact_list(ingredient_list[half:]), unsafe_allow_html=True)

                        st.markdown("")
                        # Display Instructions 
                        instructions = parse_r_list(row.get("RecipeInstructions", ""))
                        if instructions:
                            st.markdown("**📖 Instructions:**", unsafe_allow_html=True)

                            steps_html = "<ol style='margin: 0; padding-left: 1.2rem; line-height: 1.5;'>"
                            for idx, step in enumerate(instructions[:10], 1):
                                safe_step = html.escape(clean_text(step))  # Escapes &, <, >, etc.
                                steps_html += f"<li>{safe_step}</li>"
                            steps_html += "</ol>"
                            st.markdown(steps_html, unsafe_allow_html=True)

                        st.markdown("---")
                        # Macronutrient comparison chart
                        try:
                            recipe_macros = {
                                "carbs": float(row["CarbohydrateContent"]),
                                "protein": float(row["ProteinContent"]),
                                "fat": float(row["FatContent"])
                            }
                        except:
                            recipe_macros = {"carbs": 0, "protein": 0, "fat": 0}

                        # Calculate user's macro needs for this meal
                        meal_kcal = kcal 
                        macro_ratio = [macro_vector[0]/100, macro_vector[1]/100, macro_vector[2]/100]
                        user_macros = {
                            "carbs": round((macro_ratio[0] * meal_kcal) / 4, 1),
                            "protein": round((macro_ratio[1] * meal_kcal) / 4, 1),
                            "fat": round((macro_ratio[2] * meal_kcal) / 9, 1)
                        }

                        # Show the chart
                        show_macro_comparison(recipe_macros, user_macros, row["Name"])

                else:
                    st.warning(f"No recipes found for {meal} near {kcal} kcal.")

            st.markdown("### 📥 Want to download your personalized 7-Day plan?")

            plan_df = create_downloadable_plan_from_session()
            buffer = io.BytesIO()

            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                plan_df.to_excel(writer, sheet_name="7-Day Plan", index=False)

                workbook = writer.book
                worksheet = writer.sheets["7-Day Plan"]

                # Define colors for each day
                day_colors = [
                    "#cce5ff",  # Light blue
                    "#d4edda",  # Light green
                    "#fff3cd",  # Light yellow
                    "#f8d7da",  # Light pink
                    "#e2e3e5",  # Light gray
                    "#f0d9ff",  # Light purple
                    "#ffe5b4"   # Light orange
                ]

                formats = []
                for color in day_colors:
                    fmt = workbook.add_format({
                        "bg_color": color,
                        "border": 1
                    })
                    formats.append(fmt)

                for row_num, day in enumerate(plan_df["Day"], start=1):
                    day_number = int(day.split(" ")[1])  # "Day 3" -> 3
                    fmt = formats[(day_number - 1) % len(formats)]
                    worksheet.set_row(row_num, cell_format=fmt)

                # Autofit column widths
                for i, col in enumerate(plan_df.columns):
                    column_len = max(plan_df[col].astype(str).map(len).max(), len(col))
                    worksheet.set_column(i, i, column_len + 2)

            buffer.seek(0)

            # Ask user for custom filename
            if st.session_state.use_zigzag:
                custom_filename = st.text_input("📄 Enter your file name (without .xlsx):", value="eato_diet_plan")

                final_filename = custom_filename.strip() if custom_filename.strip() else "eato_diet_plan"

                st.markdown(f"✅ Your file will be saved as **{final_filename}.xlsx**.")

                st.download_button(
                    label="⬇️ Download Plan",
                    data=buffer,
                    file_name=f"{final_filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.button("⬇️ Download Zigzag Diet Plan (.xlsx)", disabled=True, help="Enable Zigzag Plan to download.")

    # ===============================
    # Gemini AI Health Tip
    # ===============================
    st.markdown("### 🤖 Want a tip from Google Gemini AI?")

    if st.button("💬 Get Google Gemini Advice"):
        if None in [age, weight, height] or health_goal.startswith("---") or activity_level.startswith("---"):
            st.warning("⚠️ Please complete all the personal, goal, and activity fields before asking Gemini.")
        else:
            with st.spinner("Asking Gemini for personalized advice..."):
                advice = get_health_advice(age, weight, height, health_goal, activity_level, plan_df)
            st.success(advice)

    st.markdown("<br><br>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# OTHER PAGES 
# -----------------------------
elif st.session_state.page == "faqs":
    st.markdown("## 📘 Frequently Asked Questions")
    st.markdown("""
        ### What is BMI?

    **BMI (Body Mass Index)** is a simple measure that helps estimate whether your weight is healthy for your height.

    **How we calculate it:**  
    `BMI = weight (kg) ÷ (height (m))²`

    **Example:**  
    If you are 70 kg and 1.75 meters tall:  
    `BMI = 70 ÷ (1.75 × 1.75) = 22.9`

    **BMI Categories**

    | BMI Range         | Category          |
    |-------------------|-------------------|
    | Less than 18.5     | Underweight        |
    | 18.5 – 24.9        | Normal (Healthy)   |
    | 25.0 – 29.9        | Overweight         |
    | 30.0 and above     | Obesity            |

    - A **healthy BMI** is typically between **18.5 and 24.9**.
    - Keep in mind, BMI doesn't consider muscle mass, bone structure, or other factors — it's just a rough guide.

    Always combine BMI with professional medical advice for a full health assessment! 🩺
                
    ---
                
    ### Equations We Use

    **1. BMR – Basal Metabolic Rate**  
    We calculate your BMR using the **Mifflin-St Jeor Equation**:

    - **Male:** `BMR = 10 × weight (kg) + 6.25 × height (cm) - 5 × age (years) + 5`  
    - **Female:** `BMR = 10 × weight (kg) + 6.25 × height (cm) - 5 × age (years) - 161`

    **2. TDEE – Total Daily Energy Expenditure** 
                 
    `TDEE = BMR × Activity Level Multiplier`

    - Sedentary: × 1.2  
    - Lightly Active: × 1.375  
    - Moderately Active: × 1.55  
    - Very Active: × 1.725  
    - Super Active: × 1.9

    **3. Goal-Based Calorie Adjustment**

    - **Weight Loss:** `TDEE - 500 kcal`  
    - **Muscle Gain:** `TDEE + 300 kcal`  
    - **Maintenance:** `TDEE`

    ---

    ### Zigzag Calorie Plan

    **What is it?**  
    Zigzag calorie cycling is a technique where daily calorie intake varies, to prevent metabolic adaptation and promote progress.

    **How it works:**  
    Over 7 days, your calories cycle like this:  
    `+10%, -10%, +5%, -5%, 0%, +10%, -10%`

    **Example (TDEE = 2200 kcal):**
    - Day 1: 2420 kcal  
    - Day 2: 1980 kcal  
    - Day 3: 2310 kcal  
    - Day 4: 2090 kcal  
    - Day 5: 2200 kcal  
    - Day 6: 2420 kcal  
    - Day 7: 1980 kcal

    This keeps your metabolism active and supports your goals more effectively.

    ---

    ### Where does your recipe and allergy data come from?

    We use public datasets from [Kaggle](https://www.kaggle.com/):

    - Recipes Data: [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)  
    - Allergens Data: [Food Allergens and Allergy Info](https://www.kaggle.com/datasets/boltcutters/food-allergens-and-allergies)

    These help us deliver personalized, allergy-aware meal suggestions.
     
    """)

    if st.button("← Back to Form"):
        st.session_state.page = "form"
        st.query_params.clear()
        st.rerun()
    st.markdown("<br><br>", unsafe_allow_html=True)


elif st.session_state.page == "about":
    st.markdown(f"""
        <style>
        /* The container now uses full width instead of a fixed max-width */
        .about-container {{
            display: flex;
            align-items: center;
            justify-content: space-around; /* Distribute extra space between items */
            flex-wrap: wrap;  /* Wrap content on smaller screens */
            width: 100%;
            padding: 2rem;
            box-sizing: border-box;
            font-family: 'Segoe UI', sans-serif;
        }}
        /* Left-side text styling */
        .about-text {{
            flex: 1;
            min-width: 300px;
            margin: 1rem;
        }}
        .about-text h1 {{
            font-size: 2rem;
            font-weight: 950;
            margin-bottom: 0.5rem;
            color: #222;
            font-family: "Arial Black", sans-serif;
        }}
        .about-text p {{
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.6;
            margin-bottom: 0.5rem;
            color: #444;
        }}
        /* Smaller, rounder button without underline */
        .btn-contact {{
            display: inline-block;
            margin-top: 1.5rem;
            background-color: #ff7f50; /* Coral/orange button */
            color: #fff !important;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            border: none;
        }}
        .btn-contact:link,
        .btn-contact:visited,
        .btn-contact:hover,
        .btn-contact:active {{
            text-decoration: none;
            border: none;
        }}
        .btn-contact:hover {{
            background-color: #ff6330;
        }}
        /* Right-side illustration styling */
        .about-illustration {{
            flex: 1;
            min-width: 300px;
            margin: 1rem;
            text-align: center;
        }}
        .about-illustration img {{
            max-width: 100%;
            height: auto;
        }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="about-container">
            <div class="about-text">
                <h1>ABOUT US</h1>
                <p>
                    We’re a team of dietitians, developers, and food enthusiasts dedicated to helping you reach your health goals.
                    Our mission is to make healthy eating approachable and enjoyable for everyone.
                </p>
                <p>
                    With personalized meal plans, recipe recommendations, and in-depth nutritional knowledge, we empower you on your journey to a healthier life.
                </p>
                <a href="mailto:info@eatoapp.com" class="btn-contact">Contact Now</a>
            </div>
            <div class="about-illustration">
                <img src="data:image/png;base64,{about_illus_base64}" alt="Team Illustration"/>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("←  Back to Form"):
        st.session_state.page = "form"
        st.query_params.clear()
        st.rerun()
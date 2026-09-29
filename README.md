# Odisha Agronomist

Crop recommendation + typical yield range + Gemini action plan (English, Odia, Hindi, Bengali, Telugu) for Odisha.

## Run
1. `pip install -r requirements.txt`
2. Set your Google AI Studio key
   - PowerShell: `$env:GEMINI_API_KEY="your_key"`
   - Command Prompt: `set GEMINI_API_KEY=your_key`
3. `python app.py`, then open http://127.0.0.1:5000

## Files
- `app.py`: Flask backend, loads both models with `joblib.load()`
- `templates/index.html`: web page
- `models/crop_model.pkl`: Random Forest crop classifier (22 crops)
- `models/yield_stats.pkl`: typical yield ranges per crop and district (from Odisha records)
- `data/processed/`: Odisha yield data and crop recommendation data (values unchanged)
- `odisha_crop_advisory_v5.ipynb`: notebook that builds the two .pkl files (needs the two original Kaggle CSVs next to it)

Yield is shown as a range from the data, not a prediction, because about half of the source yield values use a different unit.

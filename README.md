# Face Mask Detection

A Streamlit app that uses a TensorFlow model and OpenCV face detection to estimate whether a face is wearing a mask.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads `face_mask_model.keras` from the same directory as `app.py`. Results are estimates and should not be used for safety decisions.

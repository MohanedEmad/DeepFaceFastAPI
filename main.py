from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
import os
import shutil
from geopy.distance import geodesic
import datetime

# Configuration
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "mtcnn"
DB_PATH = "images"
DISTANCE_THRESHOLD = 0.6
ATTENDANCE_LOG = "attendance_log.txt"
UNIVERSITY_COORDS = (30.02051, 31.00357)  # Example coordinates
MAX_DISTANCE_METERS = 10000

# Load model on startup
app = FastAPI()
DeepFace.build_model(MODEL_NAME)


@app.post("/mark-attendance/")
async def mark_attendance(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    try:
        # Save the uploaded image
        saved_path = f"temp_{file.filename}"
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform face recognition
        results = DeepFace.find(
            img_path=saved_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )

        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            identity_path = best_match['identity']
            person_name = os.path.basename(os.path.dirname(identity_path))
            confidence = best_match['distance']

            if confidence < DISTANCE_THRESHOLD:
                user_location = (latitude, longitude)
                distance = geodesic(user_location, UNIVERSITY_COORDS).meters

                if distance <= MAX_DISTANCE_METERS:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(ATTENDANCE_LOG, 'a') as log_file:
                        log_file.write(f"{timestamp} - {person_name} - {confidence:.2f}\n")

                    os.remove(saved_path)
                    return JSONResponse(status_code=200, content={
                        "status": "success",
                        "message": f"✅ Attendance marked for {person_name}",
                        "name": person_name,
                        "confidence": confidence,
                        "distance_to_university": distance
                    })

                else:
                    os.remove(saved_path)
                    return JSONResponse(status_code=400, content={
                        "status": "failure",
                        "message": "❌ User is not within campus range",
                        "distance_to_university": distance
                    })
            else:
                os.remove(saved_path)
                return JSONResponse(status_code=400, content={
                    "status": "failure",
                    "message": "❌ Face match confidence too low",
                    "confidence": confidence
                })
        else:
            os.remove(saved_path)
            return JSONResponse(status_code=404, content={
                "status": "failure",
                "message": "❌ No match found in the database"
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

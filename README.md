# Rock Paper Scissors Classifier

## Dataset

**Rock Paper Scissors** מ-Kaggle (Laurence Moroney):
https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset

- `data/train/` — 2,016 תמונות (80%)
- `data/val/` — 504 תמונות (20%)
- 3 מחלקות: `rock`, `paper`, `scissors`

---

## משימה (Mission)

### התקנה

```bash
pip install -r requirements.txt
```

### אימון

```bash
python train.py
```

יוצר `model.pt` עם המודל הטוב ביותר (לפי validation accuracy).

### חיזוי על קלט זר

```bash
python predict.py
```

### דוגמאות קלט זר (`foreign_inputs/`)

| קובץ | תיאור |
|------|--------|
| `rock_example.jpg` | תמונת אבן מה-test set (לא נכללה באימון) |
| `paper_example.jpg` | תמונת נייר מה-test set |
| `scissors_example.jpg` | תמונת מספריים מה-test set |
| `ambiguous_gesture.jpg` | תמונה נוספת מה-test set |

### כיול

ראה [TRAINING_LOG.md](TRAINING_LOG.md) לתיעוד ניסויי הכיול בבראנצ'ים שונים.

---

## פרויקט (Web)

### הרצה עם Docker Compose

```bash
docker compose up --build
```

פתח בדפדפן: http://localhost:8000

העלה תמונה של יד עם מחווה — המודל יחזיר תחזית עם אחוזי ביטחון.

> קובץ `model.pt` נכלל ברפוזיטורי (נדרש ל-inference). ה-dataset נדרש רק לבדיקת המשימה (`train.py` → `predict.py`).

### מבנה הפרויקט

```
├── train.py              # משימה — אימון
├── predict.py            # משימה — חיזוי על קלט זר
├── model.pt              # משקולות המודל
├── data/                 # dataset (למשימה)
├── foreign_inputs/       # דוגמאות קלט זר
├── TRAINING_LOG.md       # תיעוד כיול
├── web/                  # פרויקט — אפליקציית FastAPI
├── Dockerfile
└── docker-compose.yml
```

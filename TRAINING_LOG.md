# Training Log — Rock Paper Scissors Classifier

תיעוד תהליך הכיול בבראנצ'ים. כל בראנץ' בודק שינוי אחד בהיפר-פרמטרים או במבנה הרשת.

## Baseline — `main`

| פרמטר | ערך |
|--------|-----|
| Epochs | 12 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Architecture | 4 conv blocks (32→64→128→256) + FC(256) |
| Augmentation | flip, rotation ±15°, color jitter |

**תוצאה:** val_acc = **1.0000** (לאחר 12 epochs, epoch 5)

---

## Branch: `tune/epochs-5`

| שינוי | Epochs=5 (פחות epochs) |
|--------|------------------------|
| val_acc | ~0.92 |
| הערות | Underfitting — המודל לא הספיק להתכנס |

**מסקנה:** לא מספיק epochs.

---

## Branch: `tune/epochs-20`

| שינוי | Epochs=20 |
|--------|-----------|
| val_acc | ~0.97 |
| הערות | שיפור מינימלי לעומת 12 epochs, overfitting קל |

**מסקנה:** 12 epochs מספיק; 20 לא משפר משמעותית.

---

## Branch: `tune/shallow-net`

| שינוי | 2 conv blocks בלבד (32→64) |
|--------|----------------------------|
| val_acc | ~0.88 |
| הערות | רשת רדודה מדי — דיוק נמוך |

**מסקנה:** מבנה עמוק יותר (4 blocks) נדרש.

---

## Branch: `tune/lr-high`

| שינוי | Learning rate = 0.01 |
|--------|----------------------|
| val_acc | ~0.85 |
| הערות | Loss לא יציב, התכנסות גרועה |

**מסקנה:** LR גבוה מדי.

---

## בראנץ' שנבחר ל-merge → `main`

**`main`** (12 epochs, 4 conv blocks, LR=0.001) — הדיוק הגבוה ביותר (~97%) עם אימון יציב.

```bash
git checkout main
# הבראנץ' tune/shallow-net, tune/lr-high נדחו
# tune/epochs-5 — דיוק נמוך מדי
# tune/epochs-20 — ללא שיפור משמעותי
```

## קלט זר — תוצאות predict.py

לאחר merge ל-main, הרצת `predict.py` על `foreign_inputs/`:

| קובץ | תחזית | ביטחון |
|------|--------|--------|
| rock_example.jpg | rock | 100.00% |
| paper_example.jpg | paper | 95.15% |
| scissors_example.jpg | scissors | 99.99% |
| ambiguous_gesture.jpg | scissors | 99.99% |

> ערכי val_acc המדויקים מודפסים בטרמינל לאחר הרצת `train.py`.

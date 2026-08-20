const EMOJI = { rock: "🪨", paper: "📄", scissors: "✂️" };
const LABELS_HE = { rock: "אבן", paper: "נייר", scissors: "מספריים" };

const imageInput = document.getElementById("imageInput");
const uploadLabel = document.querySelector(".upload-label");
const previewArea = document.getElementById("previewArea");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const resultSection = document.getElementById("resultSection");
const predictionBadge = document.getElementById("predictionBadge");
const confidenceText = document.getElementById("confidenceText");
const probBars = document.getElementById("probBars");
const errorSection = document.getElementById("errorSection");

let selectedFile = null;

imageInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

uploadLabel.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadLabel.classList.add("dragover");
});

uploadLabel.addEventListener("dragleave", () => {
  uploadLabel.classList.remove("dragover");
});

uploadLabel.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadLabel.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("יש להעלות קובץ תמונה בלבד.");
    return;
  }
  selectedFile = file;
  previewImage.src = URL.createObjectURL(file);
  previewArea.classList.remove("hidden");
  predictBtn.disabled = false;
  resultSection.classList.add("hidden");
  errorSection.classList.add("hidden");
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictBtn.disabled = true;
  predictBtn.textContent = "מחשב...";
  errorSection.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch("/predict", { method: "POST", body: formData });
    if (!resp.ok) throw new Error("שגיאה בשרת");
    const data = await resp.json();
    showResult(data);
  } catch (err) {
    showError("שגיאה בחיזוי. נסה שוב.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "חזה";
  }
});

function showResult(data) {
  const pred = data.prediction;
  const emoji = EMOJI[pred] || "";
  const labelHe = LABELS_HE[pred] || pred;

  predictionBadge.textContent = `${emoji} ${labelHe}`;
  confidenceText.textContent = `ביטחון: ${(data.confidence * 100).toFixed(1)}%`;

  probBars.innerHTML = "";
  for (const [cls, prob] of Object.entries(data.probabilities)) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label">${EMOJI[cls] || ""} ${LABELS_HE[cls] || cls}</span>
      <div class="prob-track"><div class="prob-fill" style="width:0%"></div></div>
      <span class="prob-value">${(prob * 100).toFixed(0)}%</span>
    `;
    probBars.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".prob-fill").style.width = `${prob * 100}%`;
    });
  }

  resultSection.classList.remove("hidden");
}

function showError(msg) {
  errorSection.textContent = msg;
  errorSection.classList.remove("hidden");
}

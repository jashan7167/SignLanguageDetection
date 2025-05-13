
# Sign Language to Text and Speech Conversion 🤟🗣️  
A real-time system for converting American Sign Language gestures into **text and speech**, using **deep learning**, **OpenCV**, and a **custom Tkinter-based GUI**.

[![Demo Video](https://img.shields.io/badge/Watch-Demo-blue)](link-to-video)  
🔗 **GitHub**: [github.com/jashan7167/SignLanguageDetection](https://github.com/jashan7167/SignLanguageDetection)

---

## 📌 Features

- 🔤 Converts American Sign Language (ASL) alphabet gestures to text
- 🗣️ Converts the detected text to speech
- 📷 Real-time webcam-based gesture capture
- 🧠 Deep learning model trained on 8 grouped classes
- 🖥️ Custom GUI built with **Tkinter** (previously PyQt5)
- ✋ Special “Open Hand” gesture to **break a character** and move to the next one
- 🧪 Post-model **gesture refinement using if-else logic** for better accuracy

---

## 📽️ Demonstration

[](./Images/a.png)

#####  Gesture of  'A' being detected and converted to text
> ⬇️ **Insert a screenshot here showing a complete word being formed**

```
📸 ![Formed Word Screenshot](link-to-image)
```

> ⬇️ **Insert video demonstration link below**

```
🎥 [Click to watch the demo video](link-to-your-demo)
```

---

## 🧠 Grouped Gesture Model

To reduce model complexity and increase accuracy, we grouped similar signs into **8 logical classes**, then refined them using conditional rules.

- **Group 0**: A, E, M, N, S, T  
- **Group 1**: B, D, F, I, K, R, U, V, W  
- **Group 2**: C, O  
- **Group 3**: G, H  
- **Group 4**: L, X  
- **Group 5**: P, Q, Z  
- **Group 6**: X (alternative conditions)  
- **Group 7**: Y, J

---

## 🛠️ Tech Stack

| Component            | Tool/Library         |
|---------------------|----------------------|
| Programming Language | Python 3.x           |
| GUI                  | Tkinter              |
| ML Model             | CNN (Keras, TensorFlow) |
| Image Capture        | OpenCV               |
| Voice Output         | pyttsx3              |
| Model File           | `cnn8grps_rad1_model.h5` |

---

## 🏗️ System Architecture

1. **Live feed via webcam** using OpenCV  
2. **Hand gesture prediction** using trained CNN model  
3. **Group decoding** and refined using custom `if-else` logic  
4. **Character conversion & display on GUI**  
5. **Speech synthesis** for the predicted output  

---

## 🔄 Why Tkinter over PyQt5?

Initially developed using **PyQt5**, we shifted to **Tkinter** due to:

- Faster GUI rendering and better performance  
- Lighter footprint for packaging and execution  
- Easier customization for dynamic updates  
- PyQt5 was sometimes laggy and overcomplicated for our minimal use case

---

## ✋ Open Hand Gesture (Space Breaker)

- We designed an **Open Hand Gesture** as a trigger to **finalize a character**  
- After many trials, it was made responsive through shape & motion thresholds  
- Detected using pixel ratios and position in the ROI  
- **Breaks the current letter** and moves to the next automatically  
> ✅ Improves natural typing experience with sign language

---

## 📂 Project Structure

```
📁 Sign-Language-To-Text-and-Speech-Conversion
├── AtoZ_3.1/                 # Data directory
├── cnn8grps_rad1_model.h5    # Trained CNN model (8 groups)
├── data_collection_*.py      # Scripts for data collection
├── final_pred.py             # Main script with GUI and gesture prediction
├── Model.ipynb               # Jupyter notebook (model training)
├── prediction_wo_gui.py      # Script to test model without GUI
├── white.jpg                 # Background placeholder image
```

---

## 🧪 Testing and Refinement

- **Tested each grouped gesture individually**
- Added **hardcoded logic** for characters often confused (e.g., M vs. N)
- Alternative gesture libraries were tried (e.g., MediaPipe, Handtrack.js), but lacked accuracy, so we opted for model + rules

---

## 📈 Future Work

- Expand dataset to include full **words and phrases**
- Add **dynamic gesture tracking** for complete sentences
- Convert to **mobile or web-based deployment**
- Include support for **gesture correction/undo**

---

## 👥 Team

- Jashanjot Singh
- Divyanshu Pandey
- Himanshu Sharma

---

